import torch
from LLM_implementation.LLM_implementation import GPTModel
from Token_utils import *
import torch.nn.functional as F
from Embedding.window import create_dataloader_v1

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,

    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
file_path = "../Embedding/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)


train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


#
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(start_context, tokenizer),
#     max_new_tokens=10,
#     context_size=GPT_CONFIG_124M["context_length"]
# )
#
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def calculate_loss(model, batch):
    input_idx = batch[:, :-1]
    target_idx = batch[:, 1:]
    logits = model(input_idx)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_idx.reshape(-1))
    return loss


def evaluate_model(model, val_loader, device, eval_iter=None):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            loss = calculate_loss(model, batch)
            total_loss += loss.item()

            if eval_iter is not None:
                break

    return total_loss / len(val_loader)


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, starting_epoch=0):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0

    for epoch in range(starting_epoch, num_epochs):
        model.train()
        train_loss = 0.0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            train_loss += loss.item()
            global_step += 1

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        if (epoch + 1) % eval_freq == 0:
            val_loss = evaluate_model(model, val_loader, device, eval_iter)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"Ep {epoch + 1:02d} | Tokens seen: {tokens_seen:,} | "
                  f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")

    return train_losses, val_losses, track_tokens_seen

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)