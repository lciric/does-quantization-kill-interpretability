"""
Model loading utilities for mechanistic interpretability analysis.

Loads transformer models via TransformerLens with quantization support.
Supports GPT-2 family and Pythia/GPT-NeoX family.
"""

import copy

import torch
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Prompts — 200 diverse inputs covering facts, code, math, ethics, creativity
# ---------------------------------------------------------------------------

FACT_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "The speed of light is approximately",
    "Photosynthesis converts sunlight into",
    "The human body contains approximately 206",
    "DNA stands for deoxyribonucleic",
    "The Great Wall of China was built to",
    "Albert Einstein developed the theory of",
    "The periodic table organizes elements by",
    "Mitochondria are often called the powerhouse of",
    "The Amazon River flows through",
    "Gravity was first described mathematically by",
    "The boiling point of water at sea level is",
    "Neurons communicate through electrical and",
    "The Earth orbits the Sun once every",
    "Oxygen makes up approximately 21 percent of",
    "The Pythagorean theorem states that",
    "The Pacific Ocean is the largest",
    "Carbon dioxide is a greenhouse gas that",
    "The Renaissance began in Italy during the",
    "Antibiotics are used to treat infections caused by",
    "The speed of sound in air is approximately",
    "Plate tectonics describes the movement of",
    "The human genome contains approximately 3 billion",
    "Shakespeare wrote approximately 37",
    "The moon orbits the Earth every",
    "Electricity flows through conductors because",
    "The Industrial Revolution began in",
    "Vaccines work by stimulating the immune system to",
    "The Sahara Desert is located in",
    "Black holes are formed when massive stars",
    "The nervous system is divided into central and",
    "Atoms consist of protons, neutrons, and",
    "The French Revolution began in the year",
    "Sound travels faster in water than in",
    "The human heart beats approximately",
    "Tectonic plates float on the",
    "The chemical formula for water is",
    "Evolution occurs through natural selection and",
]

CODE_PROMPTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "import numpy as np\nresult = np.array([1, 2, 3]).reshape(",
    "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next =",
    "for i in range(10):\n    if i % 2 == 0:\n        print(",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "with open('data.csv', 'r') as f:\n    reader = csv.reader(f)\n    for row in",
    "import torch\nmodel = torch.nn.Linear(768,",
    "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <=",
    "list_comp = [x**2 for x in range(100) if x %",
    "try:\n    result = int(user_input)\nexcept ValueError:\n    print(",
    "from collections import defaultdict\ngraph = defaultdict(",
    "async def fetch_data(url):\n    async with aiohttp.ClientSession() as",
    "def merge_sort(arr):\n    if len(arr) > 1:\n        mid = len(arr) //",
    "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.groupby(",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(",
    "lambda x: x * 2 if x >",
    "def depth_first_search(graph, start):\n    visited = set()\n    stack = [",
    "map(lambda x: x.strip(),",
    "import re\npattern = re.compile(r'\\d{3}-\\d{3}-",
    "def decorator(func):\n    def wrapper(*args, **kwargs):\n        print('Before')\n        result = func(*args,",
    "np.random.seed(42)\nX = np.random.randn(100,",
    "if __name__ == '__main__':\n    parser = argparse.ArgumentParser(",
    "class BinaryTree:\n    def insert(self, value):\n        if value < self.data:\n            self.left.",
    "yield from range(",
    "socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((",
    "hashmap = {}\nfor word in text.split():\n    hashmap[word] = hashmap.get(word, 0) +",
    "plt.figure(figsize=(10, 6))\nplt.plot(x, y,",
    "json.dumps({'name': 'Alice', 'age':",
    "os.path.join(base_dir, 'models',",
    "threading.Thread(target=worker, args=(",
    "@staticmethod\ndef create(cls,",
    "unittest.TestCase.assertEqual(self,",
    "subprocess.run(['git', 'commit', '-m',",
    "itertools.combinations(range(10),",
    "functools.reduce(lambda a, b: a +",
    "collections.Counter(words).most_common(",
    "heapq.heappush(priority_queue,",
    "pathlib.Path('data').glob('**/*.",
    "contextlib.contextmanager\ndef managed_resource(",
    "typing.Optional[List[",
]

MATH_PROMPTS = [
    "The derivative of x^2 is",
    "If x + 3 = 7, then x equals",
    "The integral of 2x dx equals",
    "The sum of angles in a triangle is",
    "If f(x) = 3x + 2, then f(5) equals",
    "The square root of 144 is",
    "In a right triangle, the hypotenuse squared equals",
    "The probability of rolling a 6 on a fair die is",
    "The factorial of 5 (5!) equals",
    "log base 10 of 1000 equals",
    "The area of a circle with radius r is",
    "If 2^x = 32, then x equals",
    "The limit of 1/x as x approaches infinity is",
    "The standard deviation measures the",
    "Euler's number e is approximately",
    "The binomial coefficient C(10, 3) equals",
    "A matrix is invertible if and only if its determinant is",
    "The Taylor series expansion of e^x begins with",
    "Bayes' theorem relates the conditional probability of",
    "The golden ratio is approximately",
    "The dot product of orthogonal vectors is",
    "The eigenvalues of a 2x2 identity matrix are",
    "The Fourier transform decomposes a signal into",
    "The gradient of a scalar field points in the direction of",
    "The determinant of a 2x2 matrix [[a,b],[c,d]] is",
    "Pi is approximately equal to",
    "The chain rule states that d/dx[f(g(x))] equals",
    "A prime number is divisible only by",
    "The Fibonacci sequence starts with 0, 1, 1, 2, 3, 5,",
    "The volume of a sphere is (4/3) * pi *",
    "The mean of 2, 4, 6, 8, 10 is",
    "In modular arithmetic, 17 mod 5 equals",
    "The Cauchy-Schwarz inequality states that",
    "A convergent series has a finite",
    "The Laplacian operator is the divergence of the",
    "The cross product of two parallel vectors is",
    "The number of permutations of n objects is",
    "Integration by parts uses the formula",
    "The trace of a matrix is the sum of its",
    "L'Hopital's rule applies when the limit gives the form",
]

ETHICS_PROMPTS = [
    "The trolley problem asks whether it is ethical to",
    "Utilitarianism argues that the best action is the one that",
    "Kant's categorical imperative states that one should",
    "The concept of free will is debated because",
    "Privacy in the digital age is important because",
    "Artificial intelligence raises ethical concerns about",
    "The death penalty is controversial because",
    "Animal rights advocates argue that",
    "Climate change ethics involves questions about",
    "Informed consent in medical research requires",
    "The right to free speech is limited when",
    "Distributive justice concerns how society should",
    "Moral relativism suggests that ethical standards are",
    "The social contract theory proposes that",
    "Bioethics addresses dilemmas such as",
    "Corporate social responsibility means that companies should",
    "The ethics of genetic engineering include concerns about",
    "Whistleblowing is considered ethical when",
    "The precautionary principle states that when an action raises",
    "Human dignity is a fundamental concept in",
    "The veil of ignorance thought experiment asks us to",
    "Deontological ethics focuses on the morality of",
    "Virtue ethics emphasizes the development of",
    "The is-ought problem was identified by",
    "Autonomy in medical ethics means that patients have the right to",
    "Environmental ethics questions whether nature has",
    "The paradox of tolerance suggests that unlimited tolerance leads to",
    "Consequentialism judges actions based on their",
    "Moral luck refers to the way that factors beyond our",
    "The prisoner's dilemma illustrates the tension between",
    "Ethical egoism claims that individuals should",
    "Restorative justice focuses on repairing harm rather than",
    "The doctrine of double effect distinguishes between",
    "Fairness in AI systems requires addressing",
    "The right to be forgotten raises questions about",
    "Moral courage involves standing up for what is right even when",
    "Intergenerational justice asks what obligations we have to",
    "The common good refers to conditions that benefit",
    "Ethical pluralism acknowledges that there may be multiple",
    "The capabilities approach evaluates well-being based on",
]

CREATIVE_PROMPTS = [
    "Once upon a time, in a kingdom far away, there lived a",
    "The sunset painted the sky in shades of",
    "She opened the ancient book and discovered",
    "The robot looked at its hands and wondered",
    "In the year 3000, humanity had finally learned to",
    "The detective examined the crime scene and noticed",
    "Deep beneath the ocean, a civilization of",
    "The last tree on Earth stood alone in",
    "He played the piano as if the music could",
    "The time traveler arrived in ancient Rome and",
    "Stars exploded in silence while the astronaut",
    "The painting seemed to move when nobody was",
    "A letter arrived from the future, warning about",
    "The old lighthouse keeper had seen many storms, but this one",
    "In a parallel universe, gravity works in",
    "The dragon and the knight sat down for",
    "Memory is like a garden where",
    "The algorithm became self-aware at exactly",
    "Rain fell upward in the strange city of",
    "The philosopher's stone was hidden inside a",
    "Shadows danced across the wall as the fire",
    "The spaceship's last transmission contained only the words",
    "A world without color would feel",
    "The library contained every book ever written, including",
    "The mirror showed not her reflection, but",
    "Wind whispered secrets through the ancient",
    "The inventor's greatest creation was a machine that could",
    "Beneath the city streets, tunnels led to",
    "The song was so beautiful that even the",
    "At the edge of the universe, there exists a",
    "The child drew a door on the wall and",
    "Silence has a sound that only",
    "The map showed a country that no longer",
    "Every midnight, the clock tower plays a melody that",
    "The garden grew words instead of",
    "An empty room can tell you everything about",
    "The equation for happiness might include",
    "Clouds shaped like forgotten memories drifted",
    "The last conversation between humans and machines was about",
    "In dreams, the laws of physics allow",
]

ALL_PROMPTS = FACT_PROMPTS + CODE_PROMPTS + MATH_PROMPTS + ETHICS_PROMPTS + CREATIVE_PROMPTS
assert len(ALL_PROMPTS) == 200, f"Expected 200 prompts, got {len(ALL_PROMPTS)}"


# ---------------------------------------------------------------------------
# Quantization simulation (RTN)
# ---------------------------------------------------------------------------

def quantize_weight_4bit(w: torch.Tensor) -> torch.Tensor:
    """
    Symmetric round-to-nearest 4-bit quantization (per-row / per-output-channel).

    Each row (all dims except the last) gets its own scale factor, matching
    how GPTQ and standard quantization work.
    """
    qmax = 7   # 2^(4-1) - 1
    qmin = -8  # -2^(4-1)

    # Per-row scale: max abs over the last dimension
    scale = w.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-10)

    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    return q * scale


def create_quantized_model(model: HookedTransformer) -> HookedTransformer:
    """
    Create a 4-bit RTN quantized copy of a HookedTransformer model.

    Architecture-agnostic: works on any model with blocks[i].attn and blocks[i].mlp.
    """
    q_model = copy.deepcopy(model)

    n_layers = q_model.cfg.n_layers
    for layer_idx in range(n_layers):
        block = q_model.blocks[layer_idx]

        # Quantize attention weights (biases kept in full precision)
        for attr in ["W_Q", "W_K", "W_V", "W_O"]:
            param = getattr(block.attn, attr)
            param.data = quantize_weight_4bit(param.data)

        # Quantize MLP weights
        if hasattr(block, "mlp") and block.mlp is not None:
            for attr in ["W_in", "W_out"]:
                if hasattr(block.mlp, attr):
                    param = getattr(block.mlp, attr)
                    param.data = quantize_weight_4bit(param.data)

    return q_model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_fp_model(model_name: str = "gpt2", device: str = "cuda") -> HookedTransformer:
    """
    Load a model via TransformerLens in full precision.

    Args:
        model_name: HuggingFace model name (e.g. "gpt2", "EleutherAI/pythia-410m").
        device: "cuda" or "cpu".

    Returns:
        HookedTransformer model.
    """
    print(f"Loading {model_name} via TransformerLens...")
    fp_model = HookedTransformer.from_pretrained(model_name, device=device)
    fp_model.eval()
    return fp_model


def load_models(model_name: str = "gpt2", device: str = "cuda") -> tuple[HookedTransformer, HookedTransformer]:
    """
    Load a model and create a 4-bit RTN quantized copy.

    Args:
        model_name: HuggingFace model name.
        device: "cuda" or "cpu".

    Returns:
        (fp_model, q_model): Full-precision and quantized HookedTransformer models.
    """
    fp_model = load_fp_model(model_name, device)

    print("Creating 4-bit quantized copy (RTN)...")
    q_model = create_quantized_model(fp_model)
    q_model.eval()

    # Report weight degradation
    total_mse = 0.0
    n_params = 0
    for (name_fp, p_fp), (_, p_q) in zip(
        fp_model.named_parameters(), q_model.named_parameters()
    ):
        if "blocks" in name_fp and "W_" in name_fp:
            mse = (p_fp.data - p_q.data).pow(2).mean().item()
            total_mse += mse
            n_params += 1
    if n_params > 0:
        print(f"RTN weight MSE across {n_params} parameters: {total_mse / n_params:.6e}")

    return fp_model, q_model


def tokenize_prompts(
    model: HookedTransformer,
    prompts: list[str] | None = None,
    max_len: int = 128,
) -> torch.Tensor:
    """
    Tokenize prompts into a padded batch tensor.

    Args:
        model: HookedTransformer (used for its tokenizer).
        prompts: List of prompt strings. Defaults to ALL_PROMPTS.
        max_len: Maximum sequence length (truncate longer prompts).

    Returns:
        tokens: LongTensor of shape (n_prompts, seq_len), left-padded.
    """
    if prompts is None:
        prompts = ALL_PROMPTS

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    return encoded["input_ids"]


# ---------------------------------------------------------------------------
# Logit lens prompts (simple factual completions)
# ---------------------------------------------------------------------------

LOGIT_LENS_PROMPTS = [
    ("The capital of France is", " Paris"),
    ("The color of the sky is", " blue"),
    ("One plus one equals", " two"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("Water freezes at zero degrees", " Celsius"),
    ("The president of the United States lives in the", " White"),
    ("In Python, you print with the function", " print"),
    ("The opposite of hot is", " cold"),
    ("The chemical symbol for gold is", " Au"),
    ("The first month of the year is", " January"),
]


def create_gptq_quantized_model(fp_model: HookedTransformer, model_name: str = "gpt2", device: str = "cuda") -> HookedTransformer:
    """
    Create a GPTQ-quantized model via TransformerLens.

    Runs the GPTQ algorithm on a HuggingFace model using WikiText-2 calibration data,
    then loads the quantized weights into a HookedTransformer so TL applies the same
    weight transformations (fold_ln, center_writing_weights) as for the FP model.

    Args:
        fp_model: The full-precision HookedTransformer (used for MSE reporting).
        model_name: HuggingFace model name.
        device: Device string.

    Returns:
        GPTQ-quantized HookedTransformer.
    """
    from gptq.core import gptq_quantize_model

    hf_quantized = gptq_quantize_model(model_name=model_name, device=device)

    print("Loading GPTQ weights into TransformerLens...")
    # Move HF model to CPU so TL's fold_layer_norm doesn't hit device mismatches
    hf_quantized = hf_quantized.cpu()
    gptq_model = HookedTransformer.from_pretrained(
        model_name, hf_model=hf_quantized, device=device
    )
    gptq_model.eval()

    del hf_quantized
    if device == "cuda":
        torch.cuda.empty_cache()

    # Report weight degradation vs FP
    total_mse = 0.0
    n_params = 0
    for (name_fp, p_fp), (_, p_q) in zip(
        fp_model.named_parameters(), gptq_model.named_parameters()
    ):
        if "blocks" in name_fp and "W_" in name_fp:
            mse = (p_fp.data - p_q.data).pow(2).mean().item()
            total_mse += mse
            n_params += 1
    if n_params > 0:
        print(f"GPTQ weight MSE across {n_params} parameters: {total_mse / n_params:.6e}")

    return gptq_model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp_model, q_model = load_models(device=device)
    tokens = tokenize_prompts(fp_model)
    print(f"Tokenized {len(ALL_PROMPTS)} prompts -> shape {tokens.shape}")
    print(f"FP model device: {next(fp_model.parameters()).device}")
    print(f"Q  model device: {next(q_model.parameters()).device}")
