from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_train", "vqav2_val"]
        elif split == "val":
            names = ["vqav2_val"]
        elif split == "test":
            names = ["vqav2_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table[index]["question_id"][question_index]

        if self.split != "test":
            answers = self.table[index]["answers"][question_index]
            labels = self.table[index]["answer_labels"][question_index]
            scores = self.table[index]["answer_scores"][question_index]
        else:
            answers = list()
            labels = list()
            scores = list()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }
