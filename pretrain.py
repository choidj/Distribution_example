import torch
import torch.nn.functional as F
from .global_variables import get_args, get_timers
from .dataset import build_train_valid_datasets
from functools import partial
from .utils import average_losses_across_data_parallel_group



def model_provider():
    """Build the model."""
    args = get_args()

    #print_rank_0
    print('Building Resnet50 model...')
    model = Resnet50(#something nessasary for the model))
    )

    return model

def train_valid_test_datasets_provider(train_val_test_samples):
    """Build train, valid, and test datasets."""

    #print_rank_0
    print("Building train, validation, and test datasets" "for VIT ...")
    train_ds, valid_ds = build_train_valid_datasets(data_path=args.data_path)
    #print_rank_0
    print("> finished creating Resnet50 datasets ...")

    return train_ds, valid_ds, None
    
def get_batch(data_iterator):
    """Build the batch."""
    data = next(data_iterator)

    # only data parallelism; no need for broadcast
    images = data[0].cuda()
    labels = data[1].cuda()

    return images, labels

def loss_func(labels, output_tensor):
    logits = output_tensor.contiguous().float()
    loss = F.cross_entropy(logits, labels)

    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    (
        images,
        labels,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Forward model. lm_labels
    output_tensor = model(images)

    return output_tensor, partial(loss_func, labels)



if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic'}
    )