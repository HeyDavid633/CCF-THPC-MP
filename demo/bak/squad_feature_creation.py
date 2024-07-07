# 6.30 为Bert的数据集SQuAD构造pkl特征文件
import pickle
from transformers.data.processors.squad import SquadV2Processor, squad_convert_examples_to_features
from transformers import BertTokenizer

# 初始化SQuAD Processor, 数据集, 和分词器
processor = SquadV2Processor()
train_examples = processor.get_train_examples('data/SQuAD')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dev_examples = processor.get_dev_examples('data/SQuAD')

print(f"Number of dev examples: {len(dev_examples)}")


# # 将SQuAD 2.0示例转换为BERT输入特征
# train_features = squad_convert_examples_to_features(
#     examples=train_examples,
#     tokenizer=tokenizer,
#     max_seq_length=384,
#     doc_stride=128,
#     max_query_length=64,
#     is_training=True,
#     return_dataset=False,
#     threads=1
# )

# dev_features = squad_convert_examples_to_features(
#     examples=dev_examples,
#     tokenizer=tokenizer,
#     max_seq_length=384,
#     doc_stride=128,
#     max_query_length=64,
#     is_training=False,  # 注意这里是False，因为我们处理的是验证集
#     return_dataset=False,
#     threads=1
# )

# # 将特征保存到磁盘上
# with open('data/SQuAD/train_features.pkl', 'wb') as f:
#     pickle.dump(train_features, f)

# with open('data/SQuAD/dev_features.pkl', 'wb') as f:
#     pickle.dump(dev_features, f)

dev_original_data = [{"id": ex.qas_id, "answers": ex.answers} for ex in dev_examples]
with open('data/SQuAD/dev_original_data.pkl', 'wb') as f:
    pickle.dump(dev_original_data, f)