"""
STEP 1 — Build the dataset.

Fine-tuning = showing the model many (input -> desired output) examples.
Here, input = a messy customer support message, output = clean JSON.

We generate the data synthetically (from templates + randomness) so you
don't need to download anything. In a real company project, this data would
come from real tickets labelled by humans (or labelled by a bigger LLM and
checked by humans).

Output: data/train.jsonl, data/valid.jsonl, data/test.jsonl
Each line looks like:
  {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "<ticket text>"},
      {"role": "assistant", "content": "<the JSON we want>"}]}
This "chat" format is what mlx_lm (and Hugging Face TRL) expect.
"""

import json
import random
from pathlib import Path

random.seed(42)  # same data every run -> reproducible experiments

SYSTEM_PROMPT = (
    "You are a support-ticket parser. Read the customer message and reply with "
    "ONLY a JSON object with these keys: "
    '"category" (one of: billing, bug, shipping, account, feature_request), '
    '"urgency" (one of: low, medium, high), '
    '"product" (one of: CloudBox, PhotoPro, TaskFlow, MailMate, or null), '
    '"order_id" (string or null), '
    '"customer_name" (string or null).'
)

PRODUCTS = ["CloudBox", "PhotoPro", "TaskFlow", "MailMate"]
NAMES = [
    "Sarah", "Ahmed", "Lucas", "Yuki", "Fatima", "John", "Priya", "Mehdi",
    "Elena", "Omar", "Chloé", "David", "Amina", "Kenji", "Laura", "Youssef",
    "Sofia", "Marco", "Nadia", "Tom",
]

# Several phrasings per category so the model learns the *concept*,
# not one sentence. {p} = product, {o} = order id.
TEMPLATES = {
    "billing": [
        "I was charged twice for {p} this month.",
        "Why did my card get billed for {p} after I cancelled?",
        "the invoice for {p} shows the wrong amount",
        "I need a refund for {p}, I was overcharged.",
        "My subscription price for {p} went up without notice.",
        "Can you explain this charge on my statement? It says {p}.",
        "payment for order {o} went through 2 times",
    ],
    "bug": [
        "{p} crashes every time I open it.",
        "When I click save in {p} nothing happens.",
        "{p} keeps freezing on the loading screen",
        "Getting an error 500 in {p} since the last update.",
        "the export button in {p} is broken",
        "{p} logs me out randomly every few minutes.",
        "Sync isn't working in {p}, files never upload.",
    ],
    "shipping": [
        "My order {o} still hasn't arrived.",
        "Where is my package? Order number {o}.",
        "order {o} says delivered but I never got it",
        "The box for order {o} arrived damaged.",
        "Can I change the delivery address for {o}?",
        "tracking for {o} hasn't updated in a week",
    ],
    "account": [
        "I can't log in to my {p} account.",
        "How do I reset my password for {p}?",
        "please delete my {p} account and all my data",
        "I want to change the email on my {p} account.",
        "My {p} account got locked after too many attempts.",
        "Two-factor codes for {p} aren't arriving.",
    ],
    "feature_request": [
        "It would be great if {p} had a dark mode.",
        "Could you add an export to PDF option in {p}?",
        "suggestion: let {p} sync with Google Calendar",
        "Please add keyboard shortcuts to {p}.",
        "Would love to see offline mode in {p} someday.",
        "Any plans to support multiple languages in {p}?",
    ],
}

# Words that signal urgency. The model must learn to read the *tone*.
URGENCY_PHRASES = {
    "high": [
        "This is URGENT!", "Please fix ASAP.", "I need this resolved today.",
        "This is blocking my whole team!!", "Critical issue, please help now.",
    ],
    "medium": [
        "Please look into it.", "Would appreciate a reply soon.",
        "Can someone check this?", "Let me know what to do.", "",
    ],
    "low": [
        "No rush.", "Whenever you get a chance.", "Not urgent at all.",
        "Just a heads up.", "Take your time.",
    ],
}

GREETINGS = ["Hi,", "Hello,", "Hey team,", "Hi there,", "Good morning,", ""]
SIGN_OFFS = ["Thanks, {n}", "Regards,\n{n}", "- {n}", "Cheers, {n}", "{n}"]


def make_order_id():
    return f"ORD-{random.randint(10000, 99999)}"


def make_example():
    category = random.choice(list(TEMPLATES))
    template = random.choice(TEMPLATES[category])

    # Feature requests are rarely urgent; make the data realistic.
    if category == "feature_request":
        urgency = random.choices(["low", "medium"], weights=[3, 1])[0]
    else:
        urgency = random.choice(["low", "medium", "high"])

    product = random.choice(PRODUCTS) if "{p}" in template else None
    order_id = make_order_id() if "{o}" in template else None
    # Sometimes mention an order id even in non-shipping tickets.
    extra_order = None
    if order_id is None and random.random() < 0.2:
        extra_order = make_order_id()
        order_id = extra_order

    name = random.choice(NAMES) if random.random() < 0.7 else None

    body = template.format(p=product or "", o=order_id or "")
    parts = [random.choice(GREETINGS), body]
    if extra_order:
        parts.append(random.choice([
            f"My order id is {extra_order}.", f"(ref {extra_order})",
            f"Order: {extra_order}",
        ]))
    parts.append(random.choice(URGENCY_PHRASES[urgency]))
    if name:
        parts.append(random.choice(SIGN_OFFS).format(n=name))
    text = " ".join(p for p in parts if p).strip()

    # A bit of noise: real users don't always use capital letters.
    if random.random() < 0.15:
        text = text.lower()

    label = {
        "category": category,
        "urgency": urgency,
        "product": product,
        "order_id": order_id,
        "customer_name": name,
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(label)},
        ]
    }


def main():
    # Avoid exact duplicates, otherwise test examples could leak into training.
    seen, examples = set(), []
    while len(examples) < 1200:
        ex = make_example()
        key = ex["messages"][1]["content"]
        if key not in seen:
            seen.add(key)
            examples.append(ex)

    random.shuffle(examples)
    splits = {
        "train": examples[:1000],   # the model learns from these
        "valid": examples[1000:1100],  # watched during training (overfitting check)
        "test": examples[1100:],    # NEVER seen in training; final score
    }

    out = Path("data")
    out.mkdir(exist_ok=True)
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):4d} examples -> data/{name}.jsonl")

    print("\nExample:")
    ex = splits["train"][0]["messages"]
    print("  INPUT :", ex[1]["content"])
    print("  OUTPUT:", ex[2]["content"])


if __name__ == "__main__":
    main()
