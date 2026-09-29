import json
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


def load_prompt_bank(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def union_bbox_xyxy(boxes: List[List[float]]) -> List[float]:
    x1 = min([b[0] for b in boxes])
    y1 = min([b[1] for b in boxes])
    x2 = max([b[2] for b in boxes])
    y2 = max([b[3] for b in boxes])
    return [float(x1), float(y1), float(x2), float(y2)]


class PromptBankSampler:
    """
    A configurable prompt sampler for OVSAM.

    Supports:
    - synonym/template sampling
    - negative prompt sampling
    - synonym consistency views

    Expected prompt_bank structure:
    {
      "single": {
        "liver": {
          "train": [...],
          "test": "..."
        },
        ...
      },
      "templates": [
        "{organ}",
        "a CT image of {organ}",
        ...
      ]
    }
    """

    def __init__(
        self,
        prompt_bank: Dict[str, Any],
        mode: str = "train",
        positive_sampling: str = "single",
        train_use_template: bool = True,
        eval_use_template: bool = False,
        p_negative: float = 0.20,
        p_consistency: float = 0.30,
        max_negatives: int = 1,
        seed: int = 3407,
    ):
        self.prompt_bank = prompt_bank
        self.mode = mode
        if positive_sampling not in {"single", "all"}:
            raise ValueError(f"Unknown positive_sampling: {positive_sampling}")
        self.positive_sampling = positive_sampling
        self.train_use_template = bool(train_use_template)
        self.eval_use_template = bool(eval_use_template)
        self.p_negative = p_negative
        self.p_consistency = p_consistency
        self.max_negatives = max_negatives
        self.rng = random.Random(seed)

        self.single_bank = prompt_bank.get("single", {})
        self.templates = prompt_bank.get("templates", ["{organ}"])

        # canonical organ names that we know how to sample
        self.single_names = set(self.single_bank.keys())

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    def _format_train_phrase(self, organ_phrase: str) -> str:
        if self.train_use_template:
            return self._sample_template(organ_phrase)
        return organ_phrase

    def _sample_template(self, organ_phrase: str) -> str:
        template = self.rng.choice(self.templates)
        return template.format(organ=organ_phrase)

    def _sample_single_train_prompt(self, canonical_name: str) -> str:
        bank = self.single_bank[canonical_name]
        phrase = self.rng.choice(bank["train"])
        return self._format_train_phrase(phrase)

    def _sample_single_consistency_pair(self, canonical_name: str) -> Optional[Tuple[str, str]]:
        bank = self.single_bank[canonical_name]
        train_prompts = bank["train"]
        if len(train_prompts) < 2:
            return None
        p1, p2 = self.rng.sample(train_prompts, 2)
        return self._format_train_phrase(p1), self._format_train_phrase(p2)

    def _build_negative_prompt_pool(self, present_names: List[str]) -> List[str]:
        all_names = list(self.single_bank.keys())
        negs = [name for name in all_names if name not in set(present_names)]
        return negs

    def _sample_negative_prompts(self, present_names: List[str]) -> List[Dict[str, str]]:
        neg_pool = self._build_negative_prompt_pool(present_names)
        if len(neg_pool) == 0:
            return []
        num = min(self.max_negatives, len(neg_pool))
        chosen = self.rng.sample(neg_pool, num)
        return [
            {
                "canonical_name": name,
                "prompt": self._sample_single_train_prompt(name),
            }
            for name in chosen
        ]

    def _build_positive_train_task(
        self,
        canonical_name: str,
        matched_instances: List[Dict[str, Any]],
        all_instances: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        primary_prompt = self._sample_single_train_prompt(canonical_name)
        aux_prompt = None
        if self.rng.random() < self.p_consistency:
            pair = self._sample_single_consistency_pair(canonical_name)
            if pair is not None:
                primary_prompt, aux_prompt = pair

        return {
            "task_type": "single",
            "canonical_name": canonical_name,
            "primary_prompt": primary_prompt,
            "aux_prompt": aux_prompt,
            "target_boxes": [inst["bbox_xyxy"] for inst in matched_instances],
            "target_category_ids": [int(inst["category_id"]) for inst in matched_instances],
            "target_instance_indices": [all_instances.index(inst) for inst in matched_instances],
        }

    # --------------------------------------------------
    # main sampling
    # --------------------------------------------------
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "train":
            return self.sample_train(sample)
        return self.sample_eval(sample)

    def sample_train(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return one training sample from one slice.
        Positive tasks are either:
            - one randomly chosen present organ (`positive_sampling=single`)
            - all present organs on this slice (`positive_sampling=all`)
        Optionally:
            negative prompt task(s)
            synonym consistency prompt pair
        """
        instances = sample["instances"]
        present_names = [x["phrase"] for x in instances]
        positive_tasks = []

        if self.positive_sampling == "single":
            chosen_inst = self.rng.choice(instances)
            positive_tasks.append(
                self._build_positive_train_task(
                    canonical_name=chosen_inst["phrase"],
                    matched_instances=[chosen_inst],
                    all_instances=instances,
                )
            )
        else:
            grouped_instances: Dict[str, List[Dict[str, Any]]] = {}
            for inst in instances:
                grouped_instances.setdefault(inst["phrase"], []).append(inst)
            for canonical_name, matched_instances in grouped_instances.items():
                positive_tasks.append(
                    self._build_positive_train_task(
                        canonical_name=canonical_name,
                        matched_instances=matched_instances,
                        all_instances=instances,
                    )
                )

        negative_prompts = []
        # Match the old task-level training semantics:
        # each positive task independently has probability p_negative to attach
        # one negative sampling event. When training is image-level with
        # positive_sampling=all, this keeps the effective negative ratio close
        # to the old TrainTaskDataset behavior instead of collapsing negatives
        # to a single Bernoulli per image.
        for _ in positive_tasks:
            if self.rng.random() < self.p_negative:
                negative_prompts.extend(self._sample_negative_prompts(present_names))

        first_positive = positive_tasks[0]

        return {
            "image": sample["image"],
            "case_id": sample["case_id"],
            "slice_z": sample["slice_z"],
            "image_path": sample["image_path"],
            "height": sample["height"],
            "width": sample["width"],
            "instances": sample["instances"],

            "positive_tasks": positive_tasks,

            # Keep legacy single-task fields for compatibility with old code paths.
            "task_type": first_positive["task_type"],
            "canonical_name": first_positive["canonical_name"],
            "primary_prompt": first_positive["primary_prompt"],
            "aux_prompt": first_positive["aux_prompt"],
            "negative_prompts": negative_prompts,
            "target_boxes": first_positive["target_boxes"],
            "target_category_ids": first_positive["target_category_ids"],
            "target_instance_indices": first_positive["target_instance_indices"],
        }

    def sample_eval(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a richer structure for evaluation. We do not collapse to a single prompt.
        The evaluator can choose:
        - canonical prompts
        - held-out synonym prompts
        - base/novel protocols
        """
        instances = sample["instances"]
        present_names = sorted(list({x["phrase"] for x in instances}))

        canonical_queries = []
        heldout_queries = []

        # single-organ eval prompts
        for organ_name in present_names:
            if organ_name not in self.single_bank:
                continue

            matched_instances = [x for x in instances if x["phrase"] == organ_name]
            boxes = [x["bbox_xyxy"] for x in matched_instances]

            canonical_queries.append({
                "query_type": "canonical",
                "canonical_name": organ_name,
                "prompt": self._sample_template(organ_name) if self.eval_use_template else organ_name,
                "target_boxes": boxes,
                "category_ids": [x["category_id"] for x in matched_instances],
            })

            heldout_prompt = self.single_bank[organ_name].get("test", None)
            if heldout_prompt is not None:
                heldout_queries.append({
                    "query_type": "heldout_synonym",
                    "canonical_name": organ_name,
                    "prompt": self._sample_template(heldout_prompt) if self.eval_use_template else heldout_prompt,
                    "target_boxes": boxes,
                    "category_ids": [x["category_id"] for x in matched_instances],
                })

        return {
            "image": sample["image"],
            "case_id": sample["case_id"],
            "slice_z": sample["slice_z"],
            "image_path": sample["image_path"],
            "height": sample["height"],
            "width": sample["width"],
            "instances": deepcopy(instances),

            "present_names": present_names,
            "canonical_queries": canonical_queries,
            "heldout_queries": heldout_queries,
        }
