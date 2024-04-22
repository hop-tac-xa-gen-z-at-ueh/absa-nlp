(absa-nlp-py3.9) (tensorml) user@2c5b3d7f-72fa-4614-a6bf-bef63499c8c5:~/absa-nlp$ python train_model.py
2024-04-22 04:51:05.011253: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-04-22 04:51:05.011309: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-04-22 04:51:05.012110: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-04-22 04:51:05.016589: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-04-22 04:51:05.866031: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
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
{'vinai/phobert-base': 256, 'vinai/phobert-large': 256}
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 388/388 [00:05<00:00, 73.77 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:00<00:00, 90.88 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:00<00:00, 90.50 examples/s]
input_ids of review 10: [0, 16, 4568, 8, 13, 205, 215, 387, 60, 124, 42, 8, 61, 19958, 183, 2538, 23, 1923, 54, 70, 62, 563, 18141, 2538, 23, 14, 2944, 99, 3544, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
256
24
3
2024-04-22 04:51:40.128853: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.150362: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.150637: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.152259: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.152548: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.152762: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.228935: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.229202: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.229440: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-04-22 04:51:40.229590: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22288 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4090, pci bus id: 0000:05:00.0, compute capability: 8.9
<_PrefetchDataset element_spec=({'input_ids': TensorSpec(shape=(None, 256), dtype=tf.int64, name=None), 'token_type_ids': TensorSpec(shape=(None, 256), dtype=tf.int64, name=None), 'attention_mask': TensorSpec(shape=(None, 256), dtype=tf.int64, name=None)}, TensorSpec(shape=(None, 40), dtype=tf.uint8, name=None))>
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
2024-04-22 04:52:10.686201: I external/local_xla/xla/service/service.cc:168] XLA service 0x7f4338f60ad0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-04-22 04:52:10.686273: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce RTX 4090, Compute Capability 8.9
2024-04-22 04:52:10.695796: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2024-04-22 04:52:10.719464: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8906
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1713761530.855155   27785 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
25/25 [==============================] - 49s 360ms/step - loss: 0.4607 - val_loss: 0.3535
Epoch 2/20
25/25 [==============================] - 4s 147ms/step - loss: 0.3516 - val_loss: 0.3336
Epoch 3/20
25/25 [==============================] - 4s 147ms/step - loss: 0.3375 - val_loss: 0.3216
Epoch 4/20
25/25 [==============================] - 4s 148ms/step - loss: 0.3250 - val_loss: 0.3173
Epoch 5/20
25/25 [==============================] - 4s 151ms/step - loss: 0.3169 - val_loss: 0.3152
Epoch 6/20
25/25 [==============================] - 4s 149ms/step - loss: 0.3095 - val_loss: 0.3135
Epoch 7/20
25/25 [==============================] - 4s 148ms/step - loss: 0.3068 - val_loss: 0.3089
Epoch 8/20
25/25 [==============================] - 4s 147ms/step - loss: 0.3026 - val_loss: 0.3045
Epoch 9/20
25/25 [==============================] - 4s 150ms/step - loss: 0.2939 - val_loss: 0.3010
Epoch 10/20
25/25 [==============================] - 4s 151ms/step - loss: 0.2858 - val_loss: 0.2967
Epoch 11/20
25/25 [==============================] - 4s 149ms/step - loss: 0.2773 - val_loss: 0.2959
Epoch 12/20
25/25 [==============================] - 4s 153ms/step - loss: 0.2645 - val_loss: 0.2929
Epoch 13/20
25/25 [==============================] - 4s 152ms/step - loss: 0.2546 - val_loss: 0.2983
Epoch 14/20
25/25 [==============================] - 4s 149ms/step - loss: 0.2470 - val_loss: 0.2987
Epoch 15/20
25/25 [==============================] - 4s 148ms/step - loss: 0.2372 - val_loss: 0.2984
Epoch 16/20
25/25 [==============================] - 4s 149ms/step - loss: 0.2275 - val_loss: 0.2988
Epoch 17/20
25/25 [==============================] - 4s 148ms/step - loss: 0.2217 - val_loss: 0.2985
Epoch 17: early stopping
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
3/3 [==============================] - 2s 59ms/step
3/3 [==============================] - 2s 59ms/step - loss: 0.3185
Example: Trò chơi này có rất nhiều tiềm năng, nhưng với những lỗi hiện tại, nó không thể chơi được. Trên điện thoại của cô ấy, trò chơi bị đứng, cô ấy phải tắt ứng dụng, còn đối với tôi, dường như cô ấy đang chơi rất tốt. Trước khi nó bị đóng băng, cô ấy đã nói với tôi rằng cô ấy đã đặt chân lên tôi khi tôi không chú ý nhưng trên màn hình của tôi, không ai đặt chân lên tôi. Nó hoàn toàn không đồng bộ trên điện thoại của cô ấy, tôi có $250 trên điện thoại của tôi, tôi chỉ còn $77. Tôi chắc chắn rằng có rất nhiều lỗi mà bạn guys đã được thông báo cho đến bây giờ. Và rất nhiều thứ khác mà các bạn có thể cải tiến. Tôi đã tạo một subreddit do cộng đồng quản lý cho trò chơi này, tôi rất mong muốn một số nhân viên của công ty bạn tham gia.
=> efficiency,negative
=> general,negative
=> reliability,negative
Report metrics
Polarity Detection
               precision    recall  f1-score   support

                  0.7922    0.9137    0.8487       313
   aesthetics     1.0000    0.0000    0.0000         6
         cost     1.0000    0.5500    0.7097        20
effectiveness     0.7273    0.5333    0.6154        15
   efficiency     0.4286    0.1875    0.2609        16
 enjoyability     0.7500    0.8182    0.7826        33
      general     0.7826    0.9730    0.8675        37
 learnability     1.0000    0.0000    0.0000         9
  reliability     0.8750    0.3500    0.5000        20
       safety     1.0000    0.0000    0.0000         7
     security     1.0000    0.0000    0.0000         4

     accuracy                         0.7875       480
    macro avg     0.8505    0.3932    0.4168       480
 weighted avg     0.7978    0.7875    0.7524       480

Polarity Detection
              precision    recall  f1-score   support

        None     0.7922    0.9137    0.8487       313
    negative     0.6164    0.4245    0.5028       106
    positive     0.4565    0.3443    0.3925        61

    accuracy                         0.7333       480
   macro avg     0.6217    0.5608    0.5813       480
weighted avg     0.7108    0.7333    0.7143       480

Aspect + Polarity
                        precision    recall  f1-score   support

       aesthetics,None     0.8750    1.0000    0.9333        42
   aesthetics,negative     1.0000    0.0000    0.0000         5
   aesthetics,positive     1.0000    0.0000    0.0000         1
             cost,None     0.7568    1.0000    0.8615        28
         cost,negative     0.8182    0.6000    0.6923        15
         cost,positive     1.0000    0.0000    0.0000         5
    effectiveness,None     0.8108    0.9091    0.8571        33
effectiveness,negative     0.4000    0.2222    0.2857         9
effectiveness,positive     0.1667    0.1667    0.1667         6
       efficiency,None     0.6829    0.8750    0.7671        32
   efficiency,negative     0.4286    0.2727    0.3333        11
   efficiency,positive     1.0000    0.0000    0.0000         5
     enjoyability,None     0.5000    0.4000    0.4444        15
 enjoyability,negative     0.4000    0.3077    0.3478        13
 enjoyability,positive     0.4615    0.6000    0.5217        20
          general,None     0.5000    0.0909    0.1538        11
      general,negative     0.6562    0.8750    0.7500        24
      general,positive     0.5714    0.6154    0.5926        13
     learnability,None     0.8125    1.0000    0.8966        39
 learnability,negative     1.0000    0.0000    0.0000         4
 learnability,positive     1.0000    0.0000    0.0000         5
      reliability,None     0.6750    0.9643    0.7941        28
  reliability,negative     0.7500    0.4000    0.5217        15
  reliability,positive     1.0000    0.0000    0.0000         5
           safety,None     0.8542    1.0000    0.9213        41
       safety,negative     1.0000    0.0000    0.0000         6
       safety,positive     1.0000    0.0000    0.0000         1
         security,None     0.9167    1.0000    0.9565        44
     security,negative     1.0000    0.0000    0.0000         4

              accuracy                         0.7333       480
             macro avg     0.7599    0.4241    0.4068       480
          weighted avg     0.7436    0.7333    0.6827       480

Summary
                    precision    recall  f1-score  support  accuracy
Aspect Detection     0.850518  0.393248  0.416789    480.0  0.787500
Polarity Detection   0.621735  0.560843  0.581327    480.0  0.733333
Aspect + Polarity    0.759878  0.424103  0.406824    480.0  0.733333
Predict on val data
3/3 [==============================] - 0s 59ms/step
3/3 [==============================] - 0s 63ms/step - loss: 0.2985
Example: Đây là trò chơi tốt nhất, nhưng ứng dụng đã không hoạt động trong nhiều tháng và tôi đã cập nhật mọi bản cập nhật mà họ cung cấp. Nhưng các bản cập nhật không bao giờ khắc phục được vấn đề và thực sự đã làm cho nó tồi tệ hơn theo thời gian. Ước gì tôi có thể chơi trò chơi này một lần nữa!
=> general,negative
Report metrics
Polarity Detection
               precision    recall  f1-score   support

                  0.7922    0.9316    0.8563       307
   aesthetics     1.0000    0.0000    0.0000         6
         cost     1.0000    0.5625    0.7200        16
effectiveness     0.4444    0.2105    0.2857        19
   efficiency     0.8571    0.3529    0.5000        17
 enjoyability     0.8158    0.8857    0.8493        35
      general     0.8913    0.9762    0.9318        42
 learnability     1.0000    0.0000    0.0000         9
  reliability     0.7000    0.3684    0.4828        19
       safety     1.0000    0.0000    0.0000         6
     security     1.0000    0.0000    0.0000         4

     accuracy                         0.8000       480
    macro avg     0.8637    0.3898    0.4205       480
 weighted avg     0.8053    0.8000    0.7633       480

Polarity Detection
              precision    recall  f1-score   support

        None     0.7922    0.9316    0.8563       307
    negative     0.7761    0.4860    0.5977       107
    positive     0.5192    0.4091    0.4576        66

    accuracy                         0.7604       480
   macro avg     0.6959    0.6089    0.6372       480
weighted avg     0.7511    0.7604    0.7438       480

Aspect + Polarity
                        precision    recall  f1-score   support

       aesthetics,None     0.8750    1.0000    0.9333        42
   aesthetics,negative     1.0000    0.0000    0.0000         2
   aesthetics,positive     1.0000    0.0000    0.0000         4
             cost,None     0.8205    1.0000    0.9014        32
         cost,negative     0.8889    0.6154    0.7273        13
         cost,positive     1.0000    0.0000    0.0000         3
    effectiveness,None     0.6154    0.8276    0.7059        29
effectiveness,negative     0.5000    0.1818    0.2667        11
effectiveness,positive     0.2000    0.1250    0.1538         8
       efficiency,None     0.7317    0.9677    0.8333        31
   efficiency,negative     0.8571    0.4615    0.6000        13
   efficiency,positive     1.0000    0.0000    0.0000         4
     enjoyability,None     0.6000    0.4615    0.5217        13
 enjoyability,negative     0.5714    0.3636    0.4444        11
 enjoyability,positive     0.6129    0.7917    0.6909        24
          general,None     0.5000    0.1667    0.2500         6
      general,negative     0.8667    0.8387    0.8525        31
      general,positive     0.4375    0.6364    0.5185        11
     learnability,None     0.8125    1.0000    0.8966        39
 learnability,negative     1.0000    0.0000    0.0000         4
 learnability,positive     1.0000    0.0000    0.0000         5
      reliability,None     0.6842    0.8966    0.7761        29
  reliability,negative     0.6000    0.4286    0.5000        14
  reliability,positive     1.0000    0.0000    0.0000         5
           safety,None     0.8750    1.0000    0.9333        42
       safety,negative     1.0000    0.0000    0.0000         5
       safety,positive     1.0000    0.0000    0.0000         1
         security,None     0.9167    1.0000    0.9565        44
     security,negative     1.0000    0.0000    0.0000         3
     security,positive     1.0000    0.0000    0.0000         1

              accuracy                         0.7604       480
             macro avg     0.7989    0.4254    0.4154       480
          weighted avg     0.7725    0.7604    0.7154       480

Summary
                    precision    recall  f1-score  support  accuracy
Aspect Detection     0.863720  0.389808  0.420536    480.0  0.800000
Polarity Detection   0.695865  0.608889  0.637205    480.0  0.760417
Aspect + Polarity    0.798850  0.425426  0.415411    480.0  0.760417
Predict on train data
25/25 [==============================] - 1s 56ms/step
25/25 [==============================] - 1s 59ms/step - loss: 0.1998
Example: Điều này cũng vui, nhưng bạn phải chuẩn bị tinh thần cho một số sự thất vọng. Hiện tại, có những lỗi ngăn cản trò chơi hoàn thành, và những lỗi này có thể xảy ra sau khi bạn đã chơi trong 30 phút hoặc hơn. Ví dụ, nếu có ai đó nợ tiền, và họ cố gắng thỏa thuận thêm để kiếm thêm tiền, điều này dường như làm dừng trò chơi. Dường như nó mất dấu vết về lượt chơi của ai (và ai vẫn còn nợ tiền). Đôi khi, nếu ai đó từ chối tất cả các thỏa thuận của bạn trong lượt chơi đó, vì một số lý do, lượt chơi của bạn kết thúc, nhưng nút Kết Thúc Lượt vẫn được hiển thị. Nhấn Kết thúc lượt ở thời điểm này sẽ dừng trò chơi. Nếu bạn đang cạn kiệt năng lượng pin và nhận được cảnh báo về pin, nó có thể ngắt kết nối bạn khỏi trò chơi. Ngoài ra, người chơi đôi khi ngừng chơi, và không có cảnh báo hoặc cách để từ chức mà không làm dừng trò chơi cho mọi người. Mặc dù có những khuyết điểm này, đây vẫn là một trò chơi thú vị, nhưng vẫn còn chỗ để cải tiến. Cá nhân tôi, tôi đang chờ những vấn đề này được giải quyết trước khi đầu tư thêm tiền vào Thẻ mùa.
=> general,negative
=> reliability,negative
Report metrics
Polarity Detection
               precision    recall  f1-score   support

                  0.7289    0.8470    0.7835      2412
   aesthetics     0.0000    0.0000    0.0000        53
         cost     0.3833    0.2930    0.3321       157
effectiveness     0.4167    0.2959    0.3460       169
   efficiency     0.3625    0.1908    0.2500       152
 enjoyability     0.7687    0.7946    0.7815       297
      general     0.8743    0.9152    0.8943       342
 learnability     0.5000    0.0471    0.0860        85
  reliability     0.3704    0.2113    0.2691       142
       safety     1.0000    0.0000    0.0000        30
     security     0.0000    0.0000    0.0000        41

     accuracy                         0.7090      3880
    macro avg     0.4913    0.3268    0.3402      3880
 weighted avg     0.6691    0.7090    0.6757      3880

Polarity Detection
              precision    recall  f1-score   support

        None     0.7289    0.8470    0.7835      2412
    negative     0.4353    0.3179    0.3675       931
    positive     0.3249    0.2402    0.2762       537

    accuracy                         0.6361      3880
   macro avg     0.4964    0.4684    0.4757      3880
weighted avg     0.6025    0.6361    0.6135      3880

Aspect + Polarity
                        precision    recall  f1-score   support

       aesthetics,None     0.8630    0.9970    0.9252       335
   aesthetics,negative     0.0000    0.0000    0.0000        24
   aesthetics,positive     1.0000    0.0000    0.0000        29
             cost,None     0.5858    0.6797    0.6293       231
         cost,negative     0.3162    0.3136    0.3149       118
         cost,positive     0.0000    0.0000    0.0000        39
    effectiveness,None     0.5560    0.6804    0.6119       219
effectiveness,negative     0.2836    0.1845    0.2235       103
effectiveness,positive     0.1132    0.0909    0.1008        66
       efficiency,None     0.6006    0.7839    0.6801       236
   efficiency,negative     0.3165    0.2315    0.2674       108
   efficiency,positive     1.0000    0.0227    0.0444        44
     enjoyability,None     0.2469    0.2198    0.2326        91
 enjoyability,negative     0.3579    0.2361    0.2845       144
 enjoyability,positive     0.4009    0.5556    0.4658       153
          general,None     0.0333    0.0217    0.0263        46
      general,negative     0.6458    0.6352    0.6405       244
      general,positive     0.3051    0.3673    0.3333        98
     learnability,None     0.7868    0.9868    0.8755       303
 learnability,negative     0.0000    0.0000    0.0000        43
 learnability,positive     0.1429    0.0238    0.0408        42
      reliability,None     0.6352    0.7927    0.7052       246
  reliability,negative     0.3250    0.2385    0.2751       109
  reliability,positive     0.0000    0.0000    0.0000        33
           safety,None     0.9227    1.0000    0.9598       358
       safety,negative     1.0000    0.0000    0.0000        21
       safety,positive     1.0000    0.0000    0.0000         9
         security,None     0.8938    0.9942    0.9413       347
     security,negative     1.0000    0.0000    0.0000        17
     security,positive     0.0000    0.0000    0.0000        24

              accuracy                         0.6361      3880
             macro avg     0.4777    0.3352    0.3193      3880
          weighted avg     0.5972    0.6361    0.5958      3880

Summary
                    precision    recall  f1-score  support  accuracy
Aspect Detection     0.491342  0.326800  0.340226   3880.0  0.709021
Polarity Detection   0.496364  0.468392  0.475738   3880.0  0.636082
Aspect + Polarity    0.477711  0.335197  0.319282   3880.0  0.636082
