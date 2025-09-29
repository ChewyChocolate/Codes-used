package com.example.capsnap

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import androidx.core.graphics.scale
import kotlin.collections.get
import kotlin.text.get

class CapsuleClassifier(context: Context) {
    private val interpreter: Interpreter

    init {
        // Load tflite model from assets
        val fileDescriptor = context.assets.openFd("capsule_efficientnet_model_plain.tflite")
        val input: MappedByteBuffer = fileDescriptor.createInputStream().channel
            .map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        interpreter = Interpreter(input)
    }

    /**
     * Run inference on a [Bitmap], returning the top label and confidence.
     * Assumes a single input tensor of shape [1, H, W, 3], float, and a
     * single output tensor of shape [1, N_CLASSES], float.
     */
    fun classify(bitmap: Bitmap): Pair<String, Float> {
        // 1) Get the expected input shape from the model
        val inputShape = interpreter.getInputTensor(0).shape() // e.g., [1, 224, 224, 3]
        val height = inputShape[1] // e.g., 224
        val width = inputShape[2]  // e.g., 224

        // 2) Resize the bitmap to the expected dimensions
        val resizedBitmap = bitmap.scale(width, height)

        // 3) Create a TensorImage with FLOAT32 data type and load the resized bitmap
        val image = TensorImage(DataType.FLOAT32)
        image.load(resizedBitmap) // Converts pixel values from [0,255] to [0,1] as FLOAT32

        // 4) Prepare output buffer
        val outputShape = interpreter.getOutputTensor(0).shape() // e.g., [1, 4]
        val outputDataType = interpreter.getOutputTensor(0).dataType() // e.g., FLOAT32
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)

        // 5) Run inference
        interpreter.run(image.buffer, outputBuffer.buffer.rewind())

        // 6) Post-process: find max confidence
        val confidences = outputBuffer.floatArray
        val maxIdx = confidences.indices.maxByOrNull { confidences[it] } ?: 0
        val labels = listOf("Appertason", "Ferix BC", "Medifortan Plus", "Mosegor", "Negative Class","Neuro Forte-E", "Pharmaton", "Renal Vitae Plus", "Skyvit")
        return labels.getOrElse(maxIdx) { "Unknown" } to confidences[maxIdx]
    }

    fun close() {
        interpreter.close()
    }
}
