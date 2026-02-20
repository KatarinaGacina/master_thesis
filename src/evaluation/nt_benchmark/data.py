from datasets import load_dataset, Dataset
from tqdm import tqdm

dataset_name = 'InstaDeepAI/nucleotide_transformer_downstream_tasks'

def get_train_dataset(task: str):
    dataset_train = load_dataset(
        path=dataset_name,
        split='train',
        streaming=True,
    )
    dataset_train = dataset_train.filter(lambda x: x['task'] == task)

    sequences = [x['sequence'] for x in tqdm(dataset_train, desc="Loading sequences")]
    labels = [x['label'] for x in tqdm(dataset_train, desc="Loading labels")]

    dataset_train_task = Dataset.from_dict({"sequence": sequences, "labels": labels})

    return dataset_train_task

def get_test_dataset(task: str):
    dataset_test = load_dataset(
        path=dataset_name,
        split='test',
        streaming=True,
    )

    dataset_test = dataset_test.filter(lambda x: x['task'] == task)

    sequences = [x['sequence'] for x in tqdm(dataset_test, desc="Loading sequences")]
    labels = [x['label'] for x in tqdm(dataset_test, desc="Loading labels")]

    dataset_test_task = Dataset.from_dict({"sequence": sequences, "labels": labels})

    return dataset_test_task