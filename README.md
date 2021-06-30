# TensorFlow Lite BERT QA Android example

This document walks through the code of a simple Android mobile application that
demonstrates
[BERT Question and Answer](https://www.tensorflow.org/lite/examples/bert_qa/overview).

## Explore the code

The app is written entirely in Java and uses the TensorFlow Lite
[Java library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/java)<!-- https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java -->
for performing BERT Question and Answer.

We're now going to walk through the most important parts of the sample code.

### Get the question and the context of the question

This mobile application gets the question and the context of the question using the functions defined in the
file
[`QaActivity.java`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/ui/QaActivity.java).


### Answerer

This BERT QA Android reference app uses the out-of-box [`BertQuestionAnswerer`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer) API from the [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer).


Inference can be done using just a few lines of code with the
[`BertQuestionAnswerer`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer)
in the TensorFlow Lite Task Library.

##### Load model and create BertQuestionAnswerer

`BertQuestionAnswerer` expects a model populated with the
[model metadata](https://www.tensorflow.org/lite/convert/metadata) and the label
file. See the
[model compatibility requirements](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer#model_compatibility_requirements)
for more details.


```java
/**
 * Load TF Lite model.
 */
 public void loadModel() {
     try {
         answerer = BertQuestionAnswerer.createFromFile(context, MODEL_PATH);
     } catch (IOException e) {
         Log.e(TAG, e.getMessage());
     }
 }
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
 /**
  * Run inference and predict the possible answers.
  */
  public List<QaAnswer> predict(String questionToAsk, String contextOfTheQuestion) {

      List<QaAnswer> apiResult = answerer.answer(contextOfTheQuestion, questionToAsk);
      return apiResult;
  }
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
private void answerQuestion(String question) {
        question = question.trim();
        if (question.isEmpty()) {
            questionEditText.setText(question);
            return;
        }

        // Append question mark '?' if not ended with '?'.
        // This aligns with question format that trains the model.
        if (!question.endsWith("?")) {
            question += '?';
        }
        final String questionToAsk = question;
        questionEditText.setText(questionToAsk);

        // Delete all pending tasks.
        handler.removeCallbacksAndMessages(null);

        // Hide keyboard and dismiss focus on text edit.
        InputMethodManager imm =
                (InputMethodManager) getSystemService(AppCompatActivity.INPUT_METHOD_SERVICE);
        imm.hideSoftInputFromWindow(getWindow().getDecorView().getWindowToken(), 0);
        View focusView = getCurrentFocus();
        if (focusView != null) {
            focusView.clearFocus();
        }

        // Reset content text view
        contentTextView.setText(content);

        questionAnswered = false;

        // Start showing Looking up snackbar
        Snackbar runningSnackbar =
                Snackbar.make(contentTextView, "Looking up answer...", Snackbar.LENGTH_INDEFINITE);
        runningSnackbar.show();

        // Run TF Lite model to get the answer.
        handler.post(
                () -> {
                    long beforeTime = System.currentTimeMillis();
                    final List<QaAnswer> answers = qaClient.predict(questionToAsk, content);
                    long afterTime = System.currentTimeMillis();
                    double totalSeconds = (afterTime - beforeTime) / 1000.0;

                    if (!answers.isEmpty()) {
                        // Get the top answer
                        QaAnswer topAnswer = answers.get(0);
                        // Dismiss the snackbar and show the answer.
                        runOnUiThread(
                                () -> {
                                    runningSnackbar.dismiss();
                                    presentAnswer(topAnswer);

                                    String displayMessage = "Top answer was successfully highlighted.";
                                    if (DISPLAY_RUNNING_TIME) {
                                        displayMessage = String.format("%s %.3fs.", displayMessage, totalSeconds);
                                    }
                                    Snackbar.make(contentTextView, displayMessage, Snackbar.LENGTH_LONG).show();
                                    questionAnswered = true;
                                });
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
