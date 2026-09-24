"""Dataset-specific prompts, answer decoding, and input formatting.

Each supported benchmark is described by a :class:`DatasetSpec`, so adding a
benchmark means adding one entry to :data:`DATASETS` rather than editing the
data-processing helpers.

The MMLU-Pro prompts reproduce, byte for byte, the ones used by the original
launch notebooks under ``tasks/launches/``: responses collected by
``tasks/launches/launch.py`` must stay comparable with the data already stored
in ``index_data/``. Their indentation and trailing spaces come from the
triple-quoted literals in those notebooks and are kept deliberately, since they
change tokenization. The CosmosQA and HellaSwag prompts share the layout but
not the stray trailing whitespace: nothing has been collected with them yet.

``input_options`` is passed to the prompt template as a list, so it is rendered
with Python list syntax. This matches the collected MMLU-Pro data and is kept
for that reason.

The RACE prompts are derived from the MMLU-Pro ones rather than written out
again: the RACE launch notebooks used byte-identical text, differing only in
the number of answer options. Deriving them keeps that identity explicit and
keeps the two prompt sets from drifting apart. RACE adds a passage block on
top, which the notebooks did not have -- they asked RACE questions without the
article, so data collected with this spec is not comparable to the indices the
notebooks produced.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

COT_REGIME = "cot"
CROPPED_REGIME = "cropped"
REGIMES = (COT_REGIME, CROPPED_REGIME)


def letter_answer_label(dataset_elem: dict) -> str:
    """Decode an ``"A"``-style answer field into a 0-based option index.

    Args:
        dataset_elem: Raw dataset row holding an uppercase letter in ``answer``.

    Returns:
        The option index as a string, e.g. ``"2"`` for ``"C"``.
    """
    return str(ord(dataset_elem["answer"]) - ord("A"))


def index_answer_label(dataset_elem: dict) -> str:
    """Decode a ``label`` field that already holds a 0-based option index.

    Handles both the string labels of HellaSwag and the integer labels of
    CosmosQA.

    Args:
        dataset_elem: Raw dataset row holding an index in ``label``.

    Returns:
        The option index as a string.
    """
    return str(dataset_elem["label"])


# The MMLU-Pro prompts below are written as explicit per-line literals rather
# than triple-quoted blocks: several lines end in a space, and one consists of
# spaces alone. That whitespace is part of what the collected data was produced
# with, it changes tokenization, and editors strip it from triple-quoted text.

MMLU_PRO_CROPPED_SYSTEM_PROMPT = (
    "\n"
    "            You are an expert at answering multiple choice questions. \n"
    "            You will be given a question and options. Follow these rules strictly:\n"
    "            \n"
    "            1. Select the correct option number between 0 and 9\n"
    "            2. Return your response as SINGLE token, only ONE number between 0 and 9\n"
    "\n"
    "            Follow this output format:\n"
    "            OPTION_NUMBER\n"
    "            \n"
    "            Example:\n"
    "            2 \n"
    "\n"
    "            Do not include any additional text except answer number between 0 and 9\n"
    "            Answer the following question:\n"
    "            "
)

MMLU_PRO_CROPPED_USER_PROMPT = (
    "\n"
    "            Question: {input_question}\n"
    "    \n"
    "            Options:\n"
    "            {input_options}\n"
    "            "
)

MMLU_PRO_COT_SYSTEM_PROMPT = (
    "\n"
    "            You are an expert in answering multiple choice questions. \n"
    "            You will be given a question and options. Follow these rules strictly:\n"
    "\n"
    "            1. Read the question and given options.\n"
    "            2. Begin a short reasoning process before giving answer and surround your reasoning with <think> and </think> tags.\n"
    "            3. After the chain of thoughts is complete, select the correct option number between 0 and 9.\n"
    "\n"
    "            Follow this output format:\n"
    "            <think>\n"
    "            SHORT_CHAIN_OF_THOUGHTS_TEXT\n"
    "            </think>\n"
    "            Answer: OPTION_NUMBER\n"
    "            \n"
    "            Example:\n"
    "            <think>\n"
    "            The capital of France is Paris, which is option 2.\n"
    "            </think>\n"
    "            Answer: 2 \n"
    "\n"
    "            Now answer the following question following output format.\n"
    "            "
)

MMLU_PRO_COT_USER_PROMPT = (
    "\n"
    "            Question: \n"
    "            {input_question}\n"
    "            \n"
    "            Options:\n"
    "            {input_options}\n"
    "            "
)

# RACE asks the same kind of question as MMLU-Pro over four options instead of
# ten, and its launch notebooks reused the MMLU-Pro prompts verbatim with that
# one substitution. The replacements below reproduce exactly that text.

RACE_CROPPED_SYSTEM_PROMPT = MMLU_PRO_CROPPED_SYSTEM_PROMPT.replace(
    "0 and 9", "0 and 3"
)

RACE_COT_SYSTEM_PROMPT = MMLU_PRO_COT_SYSTEM_PROMPT.replace("0 and 9", "0 and 3")

# The passage the notebooks never supplied. It is prepended to the unchanged
# MMLU-Pro user prompts, so the open-book and closed-book RACE conditions
# differ by this block alone.
RACE_PASSAGE_BLOCK = """
            Passage:
            {input_context}
"""

RACE_CROPPED_USER_PROMPT = RACE_PASSAGE_BLOCK + MMLU_PRO_CROPPED_USER_PROMPT

RACE_COT_USER_PROMPT = RACE_PASSAGE_BLOCK + MMLU_PRO_COT_USER_PROMPT


COSMOS_QA_CROPPED_SYSTEM_PROMPT = """
            You are an expert at answering multiple choice questions about a passage.
            You will be given a passage, a question and options. Follow these rules strictly:

            1. Select the correct option number between 0 and 3
            2. Return your response as SINGLE token, only ONE number between 0 and 3

            Follow this output format:
            OPTION_NUMBER

            Example:
            2

            Do not include any additional text except answer number between 0 and 3
            Answer the following question:
            """

COSMOS_QA_CROPPED_USER_PROMPT = """
            Passage:
            {input_context}

            Question: {input_question}

            Options:
            {input_options}
            """

COSMOS_QA_COT_SYSTEM_PROMPT = """
            You are an expert in answering multiple choice questions about a passage.
            You will be given a passage, a question and options. Follow these rules strictly:

            1. Read the passage, the question and given options.
            2. Begin a short reasoning process before giving answer and surround your reasoning with <think> and </think> tags.
            3. After the chain of thoughts is complete, select the correct option number between 0 and 3.

            Follow this output format:
            <think>
            SHORT_CHAIN_OF_THOUGHTS_TEXT
            </think>
            Answer: OPTION_NUMBER

            Example:
            <think>
            The passage says the man returned the stone, which is option 2.
            </think>
            Answer: 2

            Now answer the following question following output format.
            """

COSMOS_QA_COT_USER_PROMPT = """
            Passage:
            {input_context}

            Question:
            {input_question}

            Options:
            {input_options}
            """

HELLASWAG_CROPPED_SYSTEM_PROMPT = """
            You are an expert at choosing the most plausible continuation of a passage.
            You will be given the beginning of a passage and options. Follow these rules strictly:

            1. Select the option number that continues the passage most plausibly, between 0 and 3
            2. Return your response as SINGLE token, only ONE number between 0 and 3

            Follow this output format:
            OPTION_NUMBER

            Example:
            2

            Do not include any additional text except answer number between 0 and 3
            Continue the following passage:
            """

HELLASWAG_CROPPED_USER_PROMPT = """
            Passage beginning:
            {input_context}

            Options:
            {input_options}
            """

HELLASWAG_COT_SYSTEM_PROMPT = """
            You are an expert in choosing the most plausible continuation of a passage.
            You will be given the beginning of a passage and options. Follow these rules strictly:

            1. Read the passage beginning and given options.
            2. Begin a short reasoning process before giving answer and surround your reasoning with <think> and </think> tags.
            3. After the chain of thoughts is complete, select the option number that continues the passage most plausibly, between 0 and 3.

            Follow this output format:
            <think>
            SHORT_CHAIN_OF_THOUGHTS_TEXT
            </think>
            Answer: OPTION_NUMBER

            Example:
            <think>
            The man is holding a violin, so he most likely starts playing it, which is option 2.
            </think>
            Answer: 2

            Now continue the following passage following output format.
            """

HELLASWAG_COT_USER_PROMPT = """
            Passage beginning:
            {input_context}

            Options:
            {input_options}
            """


def _format_options(options) -> list:
    """Render answer options the way the original launch notebooks did."""
    return [f"{i}:  {option}" for i, option in enumerate(options)]


def _mmlu_pro_inputs(dataset_elem: dict) -> dict:
    """Prompt placeholders for one MMLU-Pro row."""
    return {
        "input_question": dataset_elem["question"],
        "input_options": _format_options(dataset_elem["options"]),
    }


def _cosmos_qa_inputs(dataset_elem: dict) -> dict:
    """Prompt placeholders for one CosmosQA row, including its passage."""
    return {
        "input_context": dataset_elem["context"],
        "input_question": dataset_elem["question"],
        "input_options": _format_options(
            [dataset_elem[f"answer{i}"] for i in range(4)]
        ),
    }


def _hellaswag_inputs(dataset_elem: dict) -> dict:
    """Prompt placeholders for one HellaSwag row."""
    return {
        "input_context": dataset_elem["ctx"],
        "input_options": _format_options(dataset_elem["endings"]),
    }


def _race_inputs(dataset_elem: dict) -> dict:
    """Prompt placeholders for one RACE row, including its article."""
    return {
        "input_context": dataset_elem["article"],
        "input_question": dataset_elem["question"],
        "input_options": _format_options(dataset_elem["options"]),
    }


@dataclass(frozen=True)
class DatasetSpec:
    """Everything that differs between benchmarks in one place.

    Attributes:
        name: Key used on the command line and in index filenames.
        hf_path: HuggingFace dataset path passed to ``load_dataset``.
        hf_config: Optional HuggingFace configuration name.
        hf_split: Split to collect from; must carry answer labels.
        option_count: Number of answer options per question.
        system_prompt: Regime name -> system prompt.
        user_prompt: Regime name -> user prompt template.
        build_inputs: Raw dataset row -> prompt placeholder values.
        answer_label: Raw dataset row -> expected answer token, as a string
            holding the 0-based option index.
    """

    name: str
    hf_path: str
    hf_config: Optional[str]
    hf_split: str
    option_count: int
    system_prompt: Dict[str, str]
    user_prompt: Dict[str, str]
    build_inputs: Callable[[dict], dict]
    answer_label: Callable[[dict], str]


MMLU_PRO = DatasetSpec(
    name="mmlu-pro",
    hf_path="TIGER-Lab/MMLU-Pro",
    hf_config=None,
    hf_split="test",
    option_count=10,
    system_prompt={
        CROPPED_REGIME: MMLU_PRO_CROPPED_SYSTEM_PROMPT,
        COT_REGIME: MMLU_PRO_COT_SYSTEM_PROMPT,
    },
    user_prompt={
        CROPPED_REGIME: MMLU_PRO_CROPPED_USER_PROMPT,
        COT_REGIME: MMLU_PRO_COT_USER_PROMPT,
    },
    build_inputs=_mmlu_pro_inputs,
    answer_label=letter_answer_label,
)

RACE = DatasetSpec(
    name="race",
    hf_path="ehovy/race",
    hf_config="high",
    hf_split="train",
    option_count=4,
    system_prompt={
        CROPPED_REGIME: RACE_CROPPED_SYSTEM_PROMPT,
        COT_REGIME: RACE_COT_SYSTEM_PROMPT,
    },
    user_prompt={
        CROPPED_REGIME: RACE_CROPPED_USER_PROMPT,
        COT_REGIME: RACE_COT_USER_PROMPT,
    },
    build_inputs=_race_inputs,
    answer_label=letter_answer_label,
)

# CosmosQA is loaded from a Parquet mirror rather than "allenai/cosmos_qa":
# the canonical repository ships a Python loading script, and datasets>=4.0
# refuses those outright. The mirror carries identical fields and split sizes
# (train 25262, validation 2985, test 6963).
COSMOS_QA = DatasetSpec(
    name="cosmos-qa",
    hf_path="Samsoup/cosmos_qa",
    hf_config=None,
    hf_split="train",
    option_count=4,
    system_prompt={
        CROPPED_REGIME: COSMOS_QA_CROPPED_SYSTEM_PROMPT,
        COT_REGIME: COSMOS_QA_COT_SYSTEM_PROMPT,
    },
    user_prompt={
        CROPPED_REGIME: COSMOS_QA_CROPPED_USER_PROMPT,
        COT_REGIME: COSMOS_QA_COT_USER_PROMPT,
    },
    build_inputs=_cosmos_qa_inputs,
    answer_label=index_answer_label,
)

HELLASWAG = DatasetSpec(
    name="hellaswag",
    hf_path="Rowan/hellaswag",
    hf_config=None,
    hf_split="train",
    option_count=4,
    system_prompt={
        CROPPED_REGIME: HELLASWAG_CROPPED_SYSTEM_PROMPT,
        COT_REGIME: HELLASWAG_COT_SYSTEM_PROMPT,
    },
    user_prompt={
        CROPPED_REGIME: HELLASWAG_CROPPED_USER_PROMPT,
        COT_REGIME: HELLASWAG_COT_USER_PROMPT,
    },
    build_inputs=_hellaswag_inputs,
    answer_label=index_answer_label,
)

DATASETS: Dict[str, DatasetSpec] = {
    spec.name: spec for spec in (MMLU_PRO, RACE, COSMOS_QA, HELLASWAG)
}


def get_dataset(name: str) -> DatasetSpec:
    """Look up a dataset specification by name.

    Args:
        name: Key from :data:`DATASETS`, e.g. ``"mmlu-pro"``.

    Returns:
        The matching ``DatasetSpec``.

    Raises:
        KeyError: If no dataset is registered under ``name``.
    """
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {name!r}; known datasets: {known}")
    return DATASETS[name]
