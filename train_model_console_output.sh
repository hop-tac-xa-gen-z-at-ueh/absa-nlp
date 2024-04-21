(absa-nlp-py3.9) (tensorml) user@088d75c7-c12c-4bf4-aa25-c0841d4f04e7:~/absa-nlp$ python train_model.py
2024-04-21 17:31:28.465536: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-04-21 17:31:28.465594: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-04-21 17:31:28.466589: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-04-21 17:31:28.472558: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-04-21 17:31:31.612130: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Generating train split: 388 examples [00:00, 1801.44 examples/s]
Generating val split: 48 examples [00:00, 9052.45 examples/s]
Generating test split: 48 examples [00:00, 12480.73 examples/s]
DatasetDict({
    train: Dataset({
        features: ['review', 'aesthetics', 'cost', 'effectiveness', 'efficiency', 'enjoyability', 'general', 'learnability', 'reliability', 'safety', 'security'],
        num_rows: 388
    })
    val: Dataset({
        features: ['review', 'aesthetics', 'cost', 'effectiveness', 'efficiency', 'enjoyability', 'general', 'learnability', 'reliability', 'safety', 'security'],
        num_rows: 48
    })
    test: Dataset({
        features: ['review', 'aesthetics', 'cost', 'effectiveness', 'efficiency', 'enjoyability', 'general', 'learnability', 'reliability', 'safety', 'security'],
        num_rows: 48
    })
})
(388, 10, 4)
(48, 10, 4)
(48, 10, 4)
config.json: 100%|██████████████████████████████████████████████████| 557/557 [00:00<00:00, 63.8kB/s]
vocab.txt: 100%|██████████████████████████████████████████████████| 895k/895k [00:00<00:00, 11.4MB/s]
bpe.codes: 100%|████████████████████████████████████████████████| 1.14M/1.14M [00:00<00:00, 3.11MB/s]
tokenizer.json: 100%|███████████████████████████████████████████| 3.13M/3.13M [00:00<00:00, 3.56MB/s]
{'vinai/phobert-base': 256, 'vinai/phobert-large': 256}
Map: 100%|██████████████████████████████████████████████████| 388/388 [00:05<00:00, 75.71 examples/s]
Map: 100%|████████████████████████████████████████████████████| 48/48 [00:00<00:00, 64.81 examples/s]
Map: 100%|████████████████████████████████████████████████████| 48/48 [00:00<00:00, 76.36 examples/s]
input_ids of review 10: [0, 16, 4568, 8, 13, 205, 215, 387, 60, 124, 42, 8, 61, 19958, 183, 2538, 23, 1923, 54, 70, 62, 563, 18141, 2538, 23, 14, 2944, 99, 3544, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
256
24
3
2024-04-21 17:32:12.575299: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.780493: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.780995: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.783297: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.783824: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.784269: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.885479: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.885785: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.886146: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-21 17:32:12.886313: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22288 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4090, pci bus id: 0000:05:00.0, compute capability: 8.9
<_PrefetchDataset element_spec=({'input_ids': TensorSpec(shape=(None, 256), dtype=tf.int64, name=None), 'token_type_ids': TensorSpec(shape=(None, 256), dtype=tf.int64, name=None), 'attention_mask': TensorSpec(shape=(None, 256), dtype=tf.int64, name=None)}, TensorSpec(shape=(None, 40), dtype=tf.uint8, name=None))>
tf_model.h5: 100%|████████████████████████████████████████████████| 740M/740M [00:42<00:00, 17.6MB/s]
Some layers from the model checkpoint at vinai/phobert-base were not used when initializing TFRobertaModel: ['lm_head']
- This IS expected if you are initializing TFRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
All the layers of TFRobertaModel were initialized from the model checkpoint at vinai/phobert-base.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFRobertaModel for predictions without further training.
KerasTensor(type_spec=TensorSpec(shape=(None, 3072), dtype=tf.float32, name=None), name='tf.__operators__.getitem/strided_slice:0', description="created by layer 'tf.__operators__.getitem'")
Epoch 1/20
WARNING:tensorflow:Gradients do not exist for variables ['tf_roberta_model/roberta/pooler/dense/kernel:0', 'tf_roberta_model/roberta/pooler/dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['tf_roberta_model/roberta/pooler/dense/kernel:0', 'tf_roberta_model/roberta/pooler/dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['tf_roberta_model/roberta/pooler/dense/kernel:0', 'tf_roberta_model/roberta/pooler/dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['tf_roberta_model/roberta/pooler/dense/kernel:0', 'tf_roberta_model/roberta/pooler/dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
2024-04-21 17:33:28.567770: I external/local_xla/xla/service/service.cc:168] XLA service 0x7f7d4f9b0c40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-04-21 17:33:28.567862: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce RTX 4090, Compute Capability 8.9
2024-04-21 17:33:28.578315: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2024-04-21 17:33:28.611883: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8906
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1713720808.772656   18692 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
25/25 [==============================] - 49s 370ms/step - loss: 0.4415 - val_loss: 0.3520
Epoch 2/20
25/25 [==============================] - 4s 153ms/step - loss: 0.3526 - val_loss: 0.3334
Epoch 3/20
25/25 [==============================] - 4s 152ms/step - loss: 0.3377 - val_loss: 0.3210
Epoch 4/20
25/25 [==============================] - 4s 149ms/step - loss: 0.3231 - val_loss: 0.3155
Epoch 5/20
25/25 [==============================] - 4s 151ms/step - loss: 0.3127 - val_loss: 0.3106
Epoch 6/20
25/25 [==============================] - 4s 147ms/step - loss: 0.3065 - val_loss: 0.3080
Epoch 7/20
25/25 [==============================] - 4s 147ms/step - loss: 0.3015 - val_loss: 0.3035
Epoch 8/20
25/25 [==============================] - 4s 147ms/step - loss: 0.2915 - val_loss: 0.3006
Epoch 9/20
25/25 [==============================] - 4s 147ms/step - loss: 0.2828 - val_loss: 0.2964
Epoch 10/20
25/25 [==============================] - 4s 147ms/step - loss: 0.2779 - val_loss: 0.2958
Epoch 11/20
25/25 [==============================] - 4s 146ms/step - loss: 0.2676 - val_loss: 0.2928
Epoch 12/20
25/25 [==============================] - 4s 151ms/step - loss: 0.2591 - val_loss: 0.2932
Epoch 13/20
25/25 [==============================] - 4s 150ms/step - loss: 0.2513 - val_loss: 0.2952
Epoch 14/20
25/25 [==============================] - 4s 147ms/step - loss: 0.2412 - val_loss: 0.2960
Epoch 15/20
25/25 [==============================] - 4s 149ms/step - loss: 0.2359 - val_loss: 0.2960
Epoch 16/20
25/25 [==============================] - 4s 148ms/step - loss: 0.2282 - val_loss: 0.3030
Epoch 16: early stopping
Some layers from the model checkpoint at vinai/phobert-base were not used when initializing TFRobertaModel: ['lm_head']
- This IS expected if you are initializing TFRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
All the layers of TFRobertaModel were initialized from the model checkpoint at vinai/phobert-base.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFRobertaModel for predictions without further training.
KerasTensor(type_spec=TensorSpec(shape=(None, 3072), dtype=tf.float32, name=None), name='tf.__operators__.getitem_1/strided_slice:0', description="created by layer 'tf.__operators__.getitem_1'")
Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 attention_mask (InputLayer  [(None, 256)]                0         []
 )

 input_ids (InputLayer)      [(None, 256)]                0         []

 token_type_ids (InputLayer  [(None, 256)]                0         []
 )

 tf_roberta_model_1 (TFRobe  TFBaseModelOutputWithPooli   1349982   ['attention_mask[0][0]',
 rtaModel)                   ngAndCrossAttentions(last_   72         'input_ids[0][0]',
                             hidden_state=(None, 256, 7              'token_type_ids[0][0]']
                             68),
                              pooler_output=(None, 768)
                             , past_key_values=None, hi
                             dden_states=((None, 256, 7
                             68),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768),
                              (None, 256, 768)),
                              attentions=None, cross_at
                             tentions=None)

 last_4_hidden_states (Conc  (None, 256, 3072)            0         ['tf_roberta_model_1[0][9]',
 atenate)                                                            'tf_roberta_model_1[0][10]',
                                                                     'tf_roberta_model_1[0][11]',
                                                                     'tf_roberta_model_1[0][12]']

 tf.__operators__.getitem_1  (None, 3072)                 0         ['last_4_hidden_states[0][0]']
  (SlicingOpLambda)

 dropout_75 (Dropout)        (None, 3072)                 0         ['tf.__operators__.getitem_1[0
                                                                    ][0]']

 aesthetics (Dense)          (None, 4)                    12292     ['dropout_75[0][0]']

 cost (Dense)                (None, 4)                    12292     ['dropout_75[0][0]']

 effectiveness (Dense)       (None, 4)                    12292     ['dropout_75[0][0]']

 efficiency (Dense)          (None, 4)                    12292     ['dropout_75[0][0]']

 enjoyability (Dense)        (None, 4)                    12292     ['dropout_75[0][0]']

 general (Dense)             (None, 4)                    12292     ['dropout_75[0][0]']

 learnability (Dense)        (None, 4)                    12292     ['dropout_75[0][0]']

 reliability (Dense)         (None, 4)                    12292     ['dropout_75[0][0]']

 safety (Dense)              (None, 4)                    12292     ['dropout_75[0][0]']

 security (Dense)            (None, 4)                    12292     ['dropout_75[0][0]']

 concatenate_1 (Concatenate  (None, 40)                   0         ['aesthetics[0][0]',
 )                                                                   'cost[0][0]',
                                                                     'effectiveness[0][0]',
                                                                     'efficiency[0][0]',
                                                                     'enjoyability[0][0]',
                                                                     'general[0][0]',
                                                                     'learnability[0][0]',
                                                                     'reliability[0][0]',
                                                                     'safety[0][0]',
                                                                     'security[0][0]']

==================================================================================================
Total params: 135121192 (515.45 MB)
Trainable params: 135121192 (515.45 MB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
[[0 0 0 0 2 0 0 0 0 0]
 [0 0 0 2 1 2 0 0 2 0]
 [0 1 0 0 2 1 0 0 0 0]
 [0 1 2 0 0 2 0 0 0 0]
 [0 0 0 0 0 2 0 0 0 0]
 [0 0 0 1 1 1 0 2 0 0]
 [1 0 0 1 0 1 0 1 0 0]
 [0 1 1 0 1 1 2 1 0 0]
 [2 0 1 1 2 1 1 2 0 0]
 [0 0 0 0 0 0 0 1 0 0]
 [0 0 0 1 0 1 0 1 0 0]
 [0 0 0 0 0 2 0 0 0 0]
 [0 1 0 0 1 0 0 1 0 0]
 [0 2 0 0 2 1 1 2 0 0]
 [0 0 0 0 0 0 0 2 0 0]
 [0 2 0 0 0 0 0 0 0 0]
 [0 1 0 0 1 2 0 1 1 1]
 [0 0 2 1 2 1 2 0 0 0]
 [0 0 0 0 0 1 0 0 0 0]
 [0 1 0 1 2 1 0 1 0 0]
 [1 0 1 0 0 1 2 0 1 0]
 [0 1 0 0 2 0 0 0 0 0]
 [0 2 0 0 2 2 0 0 0 0]
 [0 0 0 0 2 2 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0]
 [0 0 1 0 2 2 2 0 0 0]
 [0 0 0 0 1 0 0 0 0 0]
 [0 1 0 0 0 1 0 1 0 0]
 [1 1 2 0 1 1 1 0 0 0]
 [0 0 1 2 2 0 2 0 0 0]
 [0 1 0 0 1 1 0 0 0 0]
 [0 1 0 1 1 1 0 1 0 0]
 [0 2 0 0 0 0 0 0 0 0]
 [0 1 0 1 2 2 0 2 0 0]
 [0 1 0 2 2 1 0 1 0 0]
 [0 0 0 0 2 1 0 0 1 0]
 [0 0 0 1 2 1 0 1 1 1]
 [0 0 0 2 0 1 0 0 0 0]
 [0 0 2 1 0 1 0 0 0 0]
 [1 1 0 0 2 0 0 0 0 0]
 [0 0 2 0 2 2 0 0 0 0]
 [0 1 2 1 2 2 0 0 0 0]
 [0 2 0 0 1 0 0 1 1 1]
 [0 0 1 0 1 1 1 1 0 1]
 [0 0 1 0 1 1 0 1 1 0]
 [1 0 1 0 1 1 0 1 0 0]
 [0 0 0 0 2 2 0 0 0 0]
 [0 0 1 2 2 2 0 0 0 0]]
Predict on test data
3/3 [==============================] - 2s 57ms/step
3/3 [==============================] - 2s 55ms/step - loss: 0.3160
Example: Trò chơi này có rất nhiều tiềm năng, nhưng với những lỗi hiện tại, nó không thể chơi được. Trên điện thoại của cô ấy, trò chơi bị đứng, cô ấy phải tắt ứng dụng, còn đối với tôi, dường như cô ấy đang chơi rất tốt. Trước khi nó bị đóng băng, cô ấy đã nói với tôi rằng cô ấy đã đặt chân lên tôi khi tôi không chú ý nhưng trên màn hình của tôi, không ai đặt chân lên tôi. Nó hoàn toàn không đồng bộ trên điện thoại của cô ấy, tôi có $250 trên điện thoại của tôi, tôi chỉ còn $77. Tôi chắc chắn rằng có rất nhiều lỗi mà bạn guys đã được thông báo cho đến bây giờ. Và rất nhiều thứ khác mà các bạn có thể cải tiến. Tôi đã tạo một subreddit do cộng đồng quản lý cho trò chơi này, tôi rất mong muốn một số nhân viên của công ty bạn tham gia.
=> effectiveness,negative
=> efficiency,negative
=> enjoyability,positive
=> general,negative
=> reliability,negative
Report metrics
Polarity Detection
               precision    recall  f1-score   support

                  0.7664    0.9329    0.8415       313
   aesthetics     1.0000    0.0000    0.0000         6
         cost     1.0000    0.3500    0.5185        20
effectiveness     0.4000    0.1333    0.2000        15
   efficiency     0.5000    0.2500    0.3333        16
 enjoyability     0.8710    0.8182    0.8438        33
      general     0.7907    0.9189    0.8500        37
 learnability     1.0000    0.0000    0.0000         9
  reliability     0.8000    0.2000    0.3200        20
       safety     1.0000    0.0000    0.0000         7
     security     1.0000    0.0000    0.0000         4

     accuracy                         0.7708       480
    macro avg     0.8298    0.3276    0.3552       480
 weighted avg     0.7789    0.7708    0.7246       480

Polarity Detection
              precision    recall  f1-score   support

        None     0.7664    0.9329    0.8415       313
    negative     0.6545    0.3396    0.4472       106
    positive     0.5227    0.3770    0.4381        61

    accuracy                         0.7312       480
   macro avg     0.6479    0.5499    0.5756       480
weighted avg     0.7107    0.7312    0.7032       480

Aspect + Polarity
                        precision    recall  f1-score   support

       aesthetics,None     0.8750    1.0000    0.9333        42
   aesthetics,negative     1.0000    0.0000    0.0000         5
   aesthetics,positive     1.0000    0.0000    0.0000         1
             cost,None     0.6829    1.0000    0.8116        28
         cost,negative     1.0000    0.4667    0.6364        15
         cost,positive     1.0000    0.0000    0.0000         5
    effectiveness,None     0.6977    0.9091    0.7895        33
effectiveness,negative     0.2500    0.1111    0.1538         9
effectiveness,positive     1.0000    0.1667    0.2857         6
       efficiency,None     0.7000    0.8750    0.7778        32
   efficiency,negative     0.5000    0.3636    0.4211        11
   efficiency,positive     1.0000    0.0000    0.0000         5
     enjoyability,None     0.6471    0.7333    0.6875        15
 enjoyability,negative     0.6000    0.2308    0.3333        13
 enjoyability,positive     0.5769    0.7500    0.6522        20
          general,None     0.4000    0.1818    0.2500        11
      general,negative     0.6923    0.7500    0.7200        24
      general,positive     0.4118    0.5385    0.4667        13
     learnability,None     0.8125    1.0000    0.8966        39
 learnability,negative     1.0000    0.0000    0.0000         4
 learnability,positive     1.0000    0.0000    0.0000         5
      reliability,None     0.6279    0.9643    0.7606        28
  reliability,negative     0.6000    0.2000    0.3000        15
  reliability,positive     1.0000    0.0000    0.0000         5
           safety,None     0.8542    1.0000    0.9213        41
       safety,negative     1.0000    0.0000    0.0000         6
       safety,positive     1.0000    0.0000    0.0000         1
         security,None     0.9167    1.0000    0.9565        44
     security,negative     1.0000    0.0000    0.0000         4

              accuracy                         0.7312       480
             macro avg     0.7878    0.4221    0.4053       480
          weighted avg     0.7502    0.7312    0.6761       480

Summary
                    precision    recall  f1-score  support  accuracy
Aspect Detection     0.829825  0.327576  0.355191    480.0  0.770833
Polarity Detection   0.647892  0.549860  0.575600    480.0  0.731250
Aspect + Polarity    0.787755  0.422098  0.405304    480.0  0.731250
Predict on val data
3/3 [==============================] - 0s 55ms/step
3/3 [==============================] - 0s 59ms/step - loss: 0.3030
Example: Đây là trò chơi tốt nhất, nhưng ứng dụng đã không hoạt động trong nhiều tháng và tôi đã cập nhật mọi bản cập nhật mà họ cung cấp. Nhưng các bản cập nhật không bao giờ khắc phục được vấn đề và thực sự đã làm cho nó tồi tệ hơn theo thời gian. Ước gì tôi có thể chơi trò chơi này một lần nữa!
=> general,negative
Report metrics
Polarity Detection
               precision    recall  f1-score   support

                  0.7713    0.9446    0.8492       307
   aesthetics     1.0000    0.0000    0.0000         6
         cost     1.0000    0.4375    0.6087        16
effectiveness     0.5000    0.1053    0.1739        19
   efficiency     0.5000    0.1176    0.1905        17
 enjoyability     0.8000    0.8000    0.8000        35
      general     0.8913    0.9762    0.9318        42
 learnability     1.0000    0.0000    0.0000         9
  reliability     0.8750    0.3684    0.5185        19
       safety     1.0000    0.0000    0.0000         6
     security     1.0000    0.0000    0.0000         4

     accuracy                         0.7854       480
    macro avg     0.8489    0.3409    0.3702       480
 weighted avg     0.7872    0.7854    0.7374       480

Polarity Detection
              precision    recall  f1-score   support

        None     0.7713    0.9446    0.8492       307
    negative     0.7857    0.4112    0.5399       107
    positive     0.5417    0.3939    0.4561        66

    accuracy                         0.7500       480
   macro avg     0.6996    0.5833    0.6151       480
weighted avg     0.7429    0.7500    0.7262       480

Aspect + Polarity
                        precision    recall  f1-score   support

       aesthetics,None     0.8750    1.0000    0.9333        42
   aesthetics,negative     1.0000    0.0000    0.0000         2
   aesthetics,positive     1.0000    0.0000    0.0000         4
             cost,None     0.7805    1.0000    0.8767        32
         cost,negative     0.8571    0.4615    0.6000        13
         cost,positive     1.0000    0.0000    0.0000         3
    effectiveness,None     0.6136    0.9310    0.7397        29
effectiveness,negative     0.3333    0.0909    0.1429        11
effectiveness,positive     0.0000    0.0000    0.0000         8
       efficiency,None     0.6591    0.9355    0.7733        31
   efficiency,negative     0.5000    0.1538    0.2353        13
   efficiency,positive     1.0000    0.0000    0.0000         4
     enjoyability,None     0.4615    0.4615    0.4615        13
 enjoyability,negative     0.7143    0.4545    0.5556        11
 enjoyability,positive     0.6429    0.7500    0.6923        24
          general,None     0.5000    0.1667    0.2500         6
      general,negative     0.8889    0.7742    0.8276        31
      general,positive     0.4211    0.7273    0.5333        11
     learnability,None     0.8125    1.0000    0.8966        39
 learnability,negative     1.0000    0.0000    0.0000         4
 learnability,positive     1.0000    0.0000    0.0000         5
      reliability,None     0.7000    0.9655    0.8116        29
  reliability,negative     0.7500    0.4286    0.5455        14
  reliability,positive     1.0000    0.0000    0.0000         5
           safety,None     0.8750    1.0000    0.9333        42
       safety,negative     1.0000    0.0000    0.0000         5
       safety,positive     1.0000    0.0000    0.0000         1
         security,None     0.9167    1.0000    0.9565        44
     security,negative     1.0000    0.0000    0.0000         3
     security,positive     1.0000    0.0000    0.0000         1

              accuracy                         0.7500       480
             macro avg     0.7767    0.4100    0.3922       480
          weighted avg     0.7548    0.7500    0.6964       480

Summary
                    precision    recall  f1-score  support  accuracy
Aspect Detection     0.848871  0.340877  0.370238    480.0  0.785417
Polarity Detection   0.699553  0.583260  0.615071    480.0  0.750000
Aspect + Polarity    0.776716  0.410037  0.392168    480.0  0.750000
Predict on train data
25/25 [==============================] - 1s 54ms/step
25/25 [==============================] - 1s 59ms/step - loss: 0.2209
Example: Điều này cũng vui, nhưng bạn phải chuẩn bị tinh thần cho một số sự thất vọng. Hiện tại, có những lỗi ngăn cản trò chơi hoàn thành, và những lỗi này có thể xảy ra sau khi bạn đã chơi trong 30 phút hoặc hơn. Ví dụ, nếu có ai đó nợ tiền, và họ cố gắng thỏa thuận thêm để kiếm thêm tiền, điều này dường như làm dừng trò chơi. Dường như nó mất dấu vết về lượt chơi của ai (và ai vẫn còn nợ tiền). Đôi khi, nếu ai đó từ chối tất cả các thỏa thuận của bạn trong lượt chơi đó, vì một số lý do, lượt chơi của bạn kết thúc, nhưng nút Kết Thúc Lượt vẫn được hiển thị. Nhấn Kết thúc lượt ở thời điểm này sẽ dừng trò chơi. Nếu bạn đang cạn kiệt năng lượng pin và nhận được cảnh báo về pin, nó có thể ngắt kết nối bạn khỏi trò chơi. Ngoài ra, người chơi đôi khi ngừng chơi, và không có cảnh báo hoặc cách để từ chức mà không làm dừng trò chơi cho mọi người. Mặc dù có những khuyết điểm này, đây vẫn là một trò chơi thú vị, nhưng vẫn còn chỗ để cải tiến. Cá nhân tôi, tôi đang chờ những vấn đề này được giải quyết trước khi đầu tư thêm tiền vào Thẻ mùa.
=> effectiveness,positive
=> enjoyability,positive
=> general,negative
Report metrics
Polarity Detection
               precision    recall  f1-score   support

                  0.7285    0.8698    0.7929      2412
   aesthetics     1.0000    0.0000    0.0000        53
         cost     0.4312    0.2994    0.3534       157
effectiveness     0.4066    0.2189    0.2846       169
   efficiency     0.3699    0.1776    0.2400       152
 enjoyability     0.7710    0.7710    0.7710       297
      general     0.8792    0.9152    0.8968       342
 learnability     0.0000    0.0000    0.0000        85
  reliability     0.4648    0.2324    0.3099       142
       safety     0.0000    0.0000    0.0000        30
     security     1.0000    0.0000    0.0000        41

     accuracy                         0.7175      3880
    macro avg     0.5501    0.3168    0.3317      3880
 weighted avg     0.6803    0.7175    0.6784      3880

Polarity Detection
              precision    recall  f1-score   support

        None     0.7285    0.8698    0.7929      2412
    negative     0.4293    0.2707    0.3320       931
    positive     0.3220    0.2477    0.2800       537

    accuracy                         0.6399      3880
   macro avg     0.4933    0.4627    0.4683      3880
weighted avg     0.6004    0.6399    0.6113      3880

Aspect + Polarity
                        precision    recall  f1-score   support

       aesthetics,None     0.8634    1.0000    0.9267       335
   aesthetics,negative     1.0000    0.0000    0.0000        24
   aesthetics,positive     1.0000    0.0000    0.0000        29
             cost,None     0.6057    0.7316    0.6627       231
         cost,negative     0.3398    0.2966    0.3167       118
         cost,positive     0.0000    0.0000    0.0000        39
    effectiveness,None     0.5556    0.7534    0.6395       219
effectiveness,negative     0.1864    0.1068    0.1358       103
effectiveness,positive     0.0625    0.0303    0.0408        66
       efficiency,None     0.6032    0.8051    0.6897       236
   efficiency,negative     0.3288    0.2222    0.2652       108
   efficiency,positive     1.0000    0.0000    0.0000        44
     enjoyability,None     0.2527    0.2527    0.2527        91
 enjoyability,negative     0.3294    0.1944    0.2445       144
 enjoyability,positive     0.4151    0.5752    0.4822       153
          general,None     0.0938    0.0652    0.0769        46
      general,negative     0.6410    0.5123    0.5695       244
      general,positive     0.2671    0.4388    0.3320        98
     learnability,None     0.7798    0.9934    0.8737       303
 learnability,negative     0.0000    0.0000    0.0000        43
 learnability,positive     0.0000    0.0000    0.0000        42
      reliability,None     0.6562    0.8455    0.7389       246
  reliability,negative     0.4085    0.2661    0.3222       109
  reliability,positive     1.0000    0.0000    0.0000        33
           safety,None     0.9225    0.9972    0.9584       358
       safety,negative     1.0000    0.0000    0.0000        21
       safety,positive     0.0000    0.0000    0.0000         9
         security,None     0.8943    1.0000    0.9442       347
     security,negative     1.0000    0.0000    0.0000        17
     security,positive     1.0000    0.0000    0.0000        24

              accuracy                         0.6399      3880
             macro avg     0.5402    0.3362    0.3158      3880
          weighted avg     0.6154    0.6399    0.5950      3880

Summary
                    precision    recall  f1-score  support  accuracy
Aspect Detection     0.550106  0.316763  0.331695   3880.0  0.717526
Polarity Detection   0.493269  0.462722  0.468304   3880.0  0.639948
Aspect + Polarity    0.540190  0.336229  0.315752   3880.0  0.639948
Enter your sentence or 'No' to quit: "game hay nhưng tiền mua vật phẩm đắt quá"
=> cost,negative
Enter your sentence or 'No' to quit:
