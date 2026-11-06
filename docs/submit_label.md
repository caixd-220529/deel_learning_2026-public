# 推理结果提交说明


对于使用 YOLO 模型的同学，**强烈推荐直接使用 `src/yolo_inference.py` 脚本生成推理结果**。对于使用其他目标检测模型的同学，**请务必确保提交文件符合规定的格式要求**。

推理结果的评估程序将使用 [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) 对提交文件进行类型校验，对于不符合要求的文件，代码将直接报错，无法参与评估。具体的类型定义请参阅 `yolo_inference.py` 中的 `Detection result schema` 区域，其中通过 `BaseModel` 子类定义了推理结果的数据结构。

## 推理结果字段说明

### `ObjectPrediction` 类参数说明

```python
class ObjectPrediction(BaseModel):
    """One predicted object represented by an absolute pixel bounding box."""

    prediction_id: int
    class_id: int
    class_name: str
    confidence_score: float
    bbox_xyxy: tuple[float, float, float, float]
```

- `prediction_id`（`int`）：当前图片内的预测目标编号。脚本对每张图片从 `0` 开始依次编号；同一张图片内不重复。
- `class_id`（`int`）：预测目标的类别编号，须与评测数据集的类别表一致。启辰任务使用 `0=handwriting`、`1=tick`、`2=cross`，勤思任务使用 `0=person`.
- `class_name`（`str`）：类别名称，应与 `class_id` 对应。脚本从模型的类别名称表中读取该值，例如 `class_id=1` 对应 `class_name="tick"`。
- `confidence_score`（`float`）：模型对该预测目标的置信度，范围应为 `[0, 1]`。
- `bbox_xyxy`（`tuple[float, float, float, float]`）：目标框的四个坐标，依次为 `[xmin, ymin, xmax, ymax]`，即左上角和右下角的位置。在 JSON 中写成恰好包含四个数字的数组。坐标以原图左上角为原点，向右为 x 正方向、向下为 y 正方向，单位为原图像素，允许小数；应满足 `0 <= xmin < xmax <= width`、`0 <= ymin < ymax <= height`。请提交模型缩放、填充之前的原图坐标，不能填写归一化坐标等其他格式的坐标。

### `ImagePrediction` 类参数说明

```python
class ImagePrediction(BaseModel):
    """All detection predictions for one image."""

    image_path: str
    width: int
    height: int
    predictions: list[ObjectPrediction]
```

- `image_path`（`str`）：当前原始图片的路径，须保留原始文件名及扩展名。脚本优先保存相对于运行命令时工作目录的路径；图片位于该目录之外时，保存绝对路径。当前评估代码使用路径中的文件名匹配同名标签，因此**不要自行重命名图片**。
- `width`（`int`）：原始图片的宽度，单位为像素，须为正整数。不是模型推理时设置的 `image_size` 或缩放后的宽度。
- `height`（`int`）：原始图片的高度，单位为像素，须为正整数。`bbox_xyxy` 中所有坐标均以这里的原图宽高为参照。
- `predictions`（`list[ObjectPrediction]`）：当前图片中全部保留的预测目标，每个元素对应一个 `ObjectPrediction` 对象。脚本写入经过模型推理阈值筛选后的结果；如果没有检测到任何目标，应保留这张图片的记录并填写 `[]`，不要省略该图片，也不要填写 `null`。



### `ResultSummary` 类参数说明

```python
class ResultSummary(BaseModel):
    """Counts for a complete inference run."""

    image_count: int
    prediction_count: int
```

- `image_count`（`int`）：本次结果文件包含的图片总数，必须等于 `len(images)`。没有检测到目标、但保留了空 `predictions` 的图片也计入该数量。
- `prediction_count`（`int`）：所有图片中预测目标的总数，必须等于 `sum(len(image.predictions) for image in images)`。它不是图片数，也不是类别数。当前评估代码会核对这两个汇总值与实际记录是否一致，不一致会报错。


### `InferenceResult` 类参数说明

```python
class InferenceResult(BaseModel):
    """Versioned object-detection inference result file."""

    schema_version: Literal["1.0"] = "1.0"
    task: Literal["object_detection"] = "object_detection"
    group_id: int
    generated_at: datetime
    model: str
    coordinate_system: Literal["absolute_image_pixels"] = "absolute_image_pixels"
    summary: ResultSummary
    images: list[ImagePrediction]
```

- `schema_version`（`Literal["1.0"]`）：结果文件的格式版本，当前固定为字符串 `"1.0"`，不能填写其他内容。
- `task`（`Literal["object_detection"]`）：任务类型，固定为字符串 `"object_detection"`。
- `group_id`（`int`）：提交小组的组号。脚本从命令行参数 `--group_id` 读取，默认值为 `0`；**正式提交时请填写本组实际组号**。
- `generated_at`（`datetime`）：结果生成时间，在 JSON 中使用 ISO 8601 格式的时间字符串。脚本保存带时区的本地时间，例如 `"2026-11-06T10:00:00+08:00"`，其中 `+08:00` 表示 UTC+8。
- `model`（`str`）：生成结果所使用的模型权重路径。
- `coordinate_system`（`Literal["absolute_image_pixels"]`）：坐标系统，固定为字符串 `"absolute_image_pixels"`。
- `summary`（`ResultSummary`）：本次推理的汇总信息，在 JSON 中为包含 `image_count` 和 `prediction_count` 的对象。
- `images`（`list[ImagePrediction]`）：本次推理的逐图片结果，在 JSON 中为对象数组。应覆盖要求评测的全部图片，每张图片保留一条记录，包括没有检测到目标的图片。

### JSON 文件样例


**同学们最终需要提交的是将 `InferenceResult` 类对象序列化后得到的 JSON 文件。** 脚本默认将其命名为 `result.json`，通过 `inference_result.model_dump_json(indent=2)` 生成 UTF-8 JSON 文本。以下为包含一张图片、一个预测目标的样例。

```json
{
  "schema_version": "1.0",
  "task": "object_detection",
  "group_id": 1,
  "generated_at": "2026-11-06T10:00:00+08:00",
  "model": "/dev_data/cxd/teaching/deep_learning_2026/qichen/output/v0818/handwriting_tac-yolo26l/with_actual/ultralytics_output/weights/best.pt",
  "coordinate_system": "absolute_image_pixels",
  "summary": {
    "image_count": 1,
    "prediction_count": 1
  },
  "images": [
    {
      "image_path": "qichen/data/tac-cxd-v0818/pseudo_only/images/val/11.jpg",
      "width": 1488,
      "height": 1051,
      "predictions": [
        {
          "prediction_id": 0,
          "class_id": 0,
          "class_name": "handwriting",
          "confidence_score": 0.9698961973190308,
          "bbox_xyxy": [
            65.47350311279297,
            143.15679931640625,
            681.93017578125,
            505.7915954589844
          ]
        }
      ]
    }
  ]
}
```

同学们可以将以下代码的 `result_path` 改为自己的 JSON 文件路径后，运行以下代码，检查自己的文件是否通过 Pydantic 的类型校验。

```python
from pathlib import Path

from yolo_inference import InferenceResult


def main() -> None:
    result_path: Path = Path("result.json")
    result: InferenceResult = InferenceResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    print("Pass Pydantic validation!")


if __name__ == "__main__":
    main()
```

上述代码仅使用 Pydantic 校验字段类型、固定字符串及坐标数组长度，不会校验“`image_count` 需要等于测试集中的图片数量“等额外约束。

## 评价指标

本次课程设计采用 [mAP50-95](https://docs.ultralytics.com/guides/yolo-performance-metrics#class-wise-metrics)（mean Average Precision，平均精度均值）作为模型性能的评价指标。评价代码位于 `src/yolo_evaluation.py`，**以该代码输出的 `box_level.map` 作为最终评价分数**。

### mAP50-95

在目标检测任务中，通常使用交并比（Intersection over Union，IoU）衡量模型预测边界框与真实边界框的重合程度：

$$
\mathrm{IoU}=\frac{|B_{\mathrm{pred}}\cap B_{\mathrm{gt}}|}{|B_{\mathrm{pred}}\cup B_{\mathrm{gt}}|}
$$

其中，$B_{\mathrm{pred}}$ 和 $B_{\mathrm{gt}}$ 分别表示预测边界框和真实边界框。IoU 的取值范围为 0 到 1，数值越大，表示两个边界框的重合程度越高。

在给定 IoU 阈值的情况下，可以根据模型的检测结果计算各类别的精确率（Precision）和召回率（Recall），并绘制 Precision-Recall 曲线。该曲线下的面积称为平均精度（Average Precision，AP），对所有类别的 AP 取平均即可得到 mAP。

**mAP50-95** 进一步综合考虑不同 IoU 阈值下的检测性能。具体而言，分别计算 IoU 阈值为 0.50、0.55、0.60、……、0.95 时的 mAP，并取这 10 个结果的平均值：

$$
\mathrm{mAP50\text{-}95}=\frac{1}{10}\sum_{t\in\{0.50,0.55,\ldots,0.95\}}\mathrm{mAP}@t
$$

mAP50-95 越高，说明模型的整体检测性能越好。

## 提交说明

> **请务必提交指定格式的文件。不按格式提交的文件可能无法正常评分❗❗❗**
> 
> **请务必提交指定格式的文件。不按格式提交的文件可能无法正常评分❗❗❗**
> 
> **请务必提交指定格式的文件。不按格式提交的文件可能无法正常评分❗❗❗**

1. 同学们需要以组为单位提交 ZIP 压缩包，包含以下文件夹：
   - `qichen_val_labels`：文件夹下存放启辰数据集的标注结果。
   - `qinsi_val_labels`：文件夹下存放勤思数据集的标注结果。
2. 压缩包应为 `序号1-学号1-姓名1-序号2-学号2-姓名2-inference_result-v版本号.后缀`。如果一组只有一位同学，则压缩包的命名应该为 `序号-学号-姓名-inference_result-v版本号.后缀`。其中：
   - `序号` 为 `华南理工大学教学班点名册` 中对应列的值。
   - `学号` 必须是 12 位。
   - `版本号` 必须是大于 0 的有效十进制整数，前面加字母 `v`，例如 `v1` 或 `v2`。
   - `后缀` 只能是 `zip`，不要重复添加后缀，例如 `xxx.zip.zip`。
   - 命名示例：`1-202330459999-张三-2-202330451145-李四-inference_result-v1.zip`。
3. 如需重新提交，请依次递增版本号（如 `v2`、`v3`）。每位同学最多可提交 3 个版本，最终以最后提交的版本为准。
4. 文件名不符合上述格式或超过截止时间提交的文件，均不予接收。截止时间为 **2026 年 11 月 26 日 23:59**。
5. 请通过 `http://222.201.139.67:8080/files/inference_result_submission/` 提交文件，登录用户名为 `student`，密码为 `scut123456`。不接受通过 QQ 等其他方式提交。注意：`student` 用户的密码与提交课程作业时使用的密码不同。
6. 提交前，请检查实验报告的排版，确保格式整洁、便于批阅，并核对压缩包中的文件是否符合提交要求。实验报告的排版质量及提交文件是否符合要求，均可能影响本次作业的成绩。
7. 压缩包大小不得超过 **10 MB**，超出限制的文件将被自动删除。如有特殊情况，请联系助教。