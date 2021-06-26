# TensorFlow Lite BERT QA Android example

This document walks through the code of a simple Android mobile application that
demonstrates
[BERT Question and Answer](https://www.tensorflow.org/lite/examples/bert_qa/overview).

## Explore the code

The app is written entirely in Java and uses the TensorFlow Lite
[Java library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/java)
for performing BERT Question and Answer.

We're now going to walk through the most important parts of the sample code.

### Get the question and the context of the question

This mobile application gets the question and the context of the question using the functions defined in the
file
[`QaActivity.java`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/ui/QaActivity.java).


### Answerer

This BERT QA Android reference app uses the out-of-box API from the [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer).


#### Using the TensorFlow Lite Task Library

Inference can be done using just a few lines of code with the
[`ImageClassifier`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier)
in the TensorFlow Lite Task Library.

##### Load model and create ImageClassifier

`ImageClassifier` expects a model populated with the
[model metadata](https://www.tensorflow.org/lite/convert/metadata) and the label
file. See the
[model compatibility requirements](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier#model_compatibility_requirements)
for more details.

`ImageClassifierOptions` allows manipulation on various inference options, such
as setting the maximum number of top scored results to return using
`setMaxResults(MAX_RESULTS)`, and setting the score threshold using
`setScoreThreshold(scoreThreshold)`.

```java
// Create the ImageClassifier instance.
ImageClassifierOptions options =
    ImageClassifierOptions.builder().setMaxResults(MAX_RESULTS).build();
imageClassifier = ImageClassifier.createFromFileAndOptions(activity,
    getModelPath(), options);
```

`ImageClassifier` currently does not support configuring delegates and
multithread, but those are on our roadmap. Please stay tuned!

##### Run inference

`ImageClassifier` contains builtin logic to preprocess the input image, such as
rotating and resizing an image. Processing options can be configured through
`ImageProcessingOptions`. In the following example, input images are rotated to
the up-right angle and cropped to the center as the model expects a square input
(`224x224`). See the
[Java doc of `ImageClassifier`](https://github.com/tensorflow/tflite-support/blob/195b574f0aa9856c618b3f1ad87bd185cddeb657/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/vision/ImageProcessingOptions.java#L22)
for more details about how the underlying image processing is performed.

```java
TensorImage inputImage = TensorImage.fromBitmap(bitmap);
int width = bitmap.getWidth();
int height = bitmap.getHeight();
int cropSize = min(width, height);
ImageProcessingOptions imageOptions =
    ImageProcessingOptions.builder()
        .setOrientation(getOrientation(sensorOrientation))
        // Set the ROI to the center of the image.
        .setRoi(
            new Rect(
                /*left=*/ (width - cropSize) / 2,
                /*top=*/ (height - cropSize) / 2,
                /*right=*/ (width + cropSize) / 2,
                /*bottom=*/ (height + cropSize) / 2))
        .build();

List<Classifications> results = imageClassifier.classify(inputImage,
    imageOptions);
```

The output of `ImageClassifier` is a list of `Classifications` instance, where
each `Classifications` element is a single head classification result. All the
demo models are single head models, therefore, `results` only contains one
`Classifications` object. Use `Classifications.getCategories()` to get a list of
top-k categories as specified with `MAX_RESULTS`. Each `Category` object
contains the srting label and the score of that category.

To match the implementation of
[`lib_support`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support),
`results` is converted into `List<Recognition>` in the method,
`getRecognitions`.

##### Recognize image

Rather than call `run` directly, the method `recognizeImage` is used. It accepts
a bitmap and sensor orientation, runs inference, and returns a sorted `List` of
`Recognition` instances, each corresponding to a label. The method will return a
number of results bounded by `MAX_RESULTS`, which is 3 by default.

`Recognition` is a simple class that contains information about a specific
recognition result, including its `title` and `confidence`. Using the
post-processing normalization method specified, the confidence is converted to
between 0 and 1 of a given class being represented by the image.

```java
/** Gets the label to probability map. */
Map<String, Float> labeledProbability =
    new TensorLabel(labels,
        probabilityProcessor.process(outputProbabilityBuffer))
        .getMapWithFloatValue();
```

A `PriorityQueue` is used for sorting.

```java
/** Gets the top-k results. */
private static List<Recognition> getTopKProbability(
    Map<String, Float> labelProb) {
  // Find the best classifications.
  PriorityQueue<Recognition> pq =
      new PriorityQueue<>(
          MAX_RESULTS,
          new Comparator<Recognition>() {
            @Override
            public int compare(Recognition lhs, Recognition rhs) {
              // Intentionally reversed to put high confidence at the head of
              // the queue.
              return Float.compare(rhs.getConfidence(), lhs.getConfidence());
            }
          });

  for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
    pq.add(new Recognition("" + entry.getKey(), entry.getKey(),
               entry.getValue(), null));
  }

  final ArrayList<Recognition> recognitions = new ArrayList<>();
  int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
  for (int i = 0; i < recognitionsSize; ++i) {
    recognitions.add(pq.poll());
  }
  return recognitions;
}
```

### Display results

The classifier is invoked and inference results are displayed by the
`processImage()` function in
[`ClassifierActivity.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/ClassifierActivity.java).

`ClassifierActivity` is a subclass of `CameraActivity` that contains method
implementations that render the camera image, run classification, and display
the results. The method `processImage()` runs classification on a background
thread as fast as possible, rendering information on the UI thread to avoid
blocking inference and creating latency.

```java
@Override
protected void processImage() {
  rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth,
      previewHeight);
  final int imageSizeX = classifier.getImageSizeX();
  final int imageSizeY = classifier.getImageSizeY();

  runInBackground(
      new Runnable() {
        @Override
        public void run() {
          if (classifier != null) {
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results =
                classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            LOGGER.v("Detect: %s", results);

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showResultsInBottomSheet(results);
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(imageSizeX + "x" + imageSizeY);
                    showCameraResolution(imageSizeX + "x" + imageSizeY);
                    showRotationInfo(String.valueOf(sensorOrientation));
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
          readyForNextImage();
        }
      });
}
```

Another important role of `ClassifierActivity` is to determine user preferences
(by interrogating `CameraActivity`), and instantiate the appropriately
configured `Classifier` subclass. This happens when the video feed begins (via
`onPreviewSizeChosen()`) and when options are changed in the UI (via
`onInferenceConfigurationChanged()`).

```java
private void recreateClassifier(Model model, Device device, int numThreads) {
  if (classifier != null) {
    LOGGER.d("Closing classifier.");
    classifier.close();
    classifier = null;
  }
  if (device == Device.GPU && model == Model.QUANTIZED) {
    LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
    runOnUiThread(
        () -> {
          Toast.makeText(this, "GPU does not yet supported quantized models.",
              Toast.LENGTH_LONG)
              .show();
        });
    return;
  }
  try {
    LOGGER.d(
        "Creating classifier (model=%s, device=%s, numThreads=%d)", model,
        device, numThreads);
    classifier = Classifier.create(this, model, device, numThreads);
  } catch (IOException e) {
    LOGGER.e(e, "Failed to create classifier.");
  }
}
```
