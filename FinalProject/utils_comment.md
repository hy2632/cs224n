# util.py 注释

1. **class SQuAD**: 处理 data.dataset，将字典转化到类变量
2. **def collate_fn(examples) -> (lists)**
   1. 将 examples: list(tuple(5)) 分为 tuple(5 lists)的技巧 zip(\*examples)
      ```python
      example1 = (1, "str", [3])
      example2 = (2, "str2", [100])
      examples = (example1, example2)
      ints, strs, lists = zip(*examples)
      ints, strs, lists
      -----------------
      ((1, 2), ('str', 'str2'), ([3], [100]))
      ```
   2. 输入是一堆 example(tuple)的 list，输出是 5 个 list 的 tuple。
      ```python
      (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)
      ```
3. **class AverageMeter**
   平均值记录和更新，包含 avg, sum, count

4. **class EMA(model, decay)**

   ```python
   self.decay = decay
   self.shadow = {}
   self.original = {}
   ```

   1. shadow 存放需要梯度的参数
   2. model.named_parameters(): 可转换为 dict: {name: value}
   3. assign 负责更新（也备份到 self.original），resume 负责用 self.original 进行恢复

5. **class CheckpointSaver**

   1. 功能：save & load model checkpoints: 保存最好的 ckpt, 依据 metric value, max_checkpoints 达到时覆盖
   2. self.log (logging.Logger) 用法
   3. ```python
      model.__class__.__name__
      ```
      读取类名称。

6. **def load_model**
   功能: load model parameters from disk

7. **def get_available_devices**
   get IDs of all available GPUs

8. **def masked_softmax**

   1. 功能： 对`logits` 的某 dimension softmax / log_softmax. 若`mask`则设置为 0.
   2. 参数：
      ```python
      """
      Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
      """
      ```

9. **def visualize**
   在 TensorBoard 中绘制。需指定 json 文件路径

10. **def save_preds**
    将 preds: [(id, start, end)...] 按 id 排序，存储为 csv

    ```python
    np.savetxt(save_path, np.array(preds), delimiter=',', fmt='%d')
    ```

11. **def get_save_dir**
    每次 training 后保存，新建一个文件夹并序号变为 uid。uid 遍历检查是否路径名重复。
    **`name` 就是 python train.py -n {`name`} 指定的。会在`/save/`下新建子目录。**

12. **def get_logger**
    和 **logger** 相关，暂时没看懂。

13. **def torch_from_json(path, dtype=torch.float32)**
    从JSON文件加载torch.Tensor。
    先用np读取再转换。
    读取json文件一般方法：
    ```python
    with open(path, 'r') as fh:
        _ = json.load(fh)
    ...
    ```

14. **def discretize(p_start, p_end, max_len=15, no_answer=False)**
    