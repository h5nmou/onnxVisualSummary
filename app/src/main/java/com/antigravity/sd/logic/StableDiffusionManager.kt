package com.antigravity.sd.logic

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Half
import android.util.Log
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.Collections
import java.util.EnumSet
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

class StableDiffusionManager(private val context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var tokenizer = CLIPTokenizer()

    // Sessions
    private var textEncoderSession: OrtSession? = null
    private var unetSession: OrtSession? = null
    private var vaeDecoderSession: OrtSession? = null

    // Scheduler Config
    private val numInferenceSteps = 20
    private val betaStart = 0.00085f
    private val betaEnd = 0.012f
    private val betas = FloatArray(1000)
    private val alphas = FloatArray(1000)
    private val alphasCumprod = FloatArray(1000)
    private val sigmas = FloatArray(1000)
    
    init {
        // Initialize Scheduler Constants (Linear Beta Schedule)
        var alphaCumprodPrev = 1.0f
        for (i in 0 until 1000) {
            val beta = betaStart + (betaEnd - betaStart) * i / 999f
            betas[i] = beta
            alphas[i] = 1.0f - beta
            alphasCumprod[i] = alphas[i] * alphaCumprodPrev
            alphaCumprodPrev = alphasCumprod[i]
            sigmas[i] = sqrt((1 - alphasCumprod[i]) / alphasCumprod[i])
        }
    }

    // Track Active Device
    private var activeDevice = "CPU"

    // Track Active Device


    suspend fun initializeModels(onStatus: (String) -> Unit) = withContext(Dispatchers.IO) {
        try {
            onStatus("Installing Models...")
            Log.d("Antigravity-SD", "Installing Models...")

            // 1. Setup NPU/CPU Options (Moved up for validation use)
            val npuPropeties = OrtSession.SessionOptions()
            npuPropeties.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT) // LOW OPT for Stability
            npuPropeties.setSessionLogVerbosityLevel(0)
            npuPropeties.addConfigEntry("session.logid", "S25_Inference_Diagnostic")
            // REMOVED: use_ort_model_bytes_directly (causes AHardwareBuffer crash on large models)
            npuPropeties.setInterOpNumThreads(1)
            npuPropeties.setIntraOpNumThreads(1) 
            
            Log.d("Antigravity-SD", "Attempting FORCE NPU Load...")
            try {
                // Force NNAPI with FP16
                val flags = EnumSet.of(NNAPIFlags.USE_FP16)
                npuPropeties.addNnapi(flags)
                activeDevice = "NPU_FORCE_TEST"
                Log.d("Antigravity-SD", "NNAPI Provider enabled (BASIC_OPT + FP16).")
            } catch (e: Exception) {
                Log.e("Antigravity-SD", "CRITICAL NPU FAILURE: Force Mode prevented CPU fallback.", e)
                throw e // CRASH APP to see native error
            }

            val cpuOptions = OrtSession.SessionOptions()
            cpuOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            cpuOptions.addConfigEntry("session.use_ort_model_bytes_directly", "1")
            cpuOptions.setInterOpNumThreads(1)
            cpuOptions.setIntraOpNumThreads(4)

            // 2. Smart Asset Search (Scan & Validate)
            var assetPrefix = "model/sd/" // Fallback
            val candidates = listOf("models/sd", "model/sd", "sd", "")
            var foundValidTextEncoder = false
            
            for (dir in candidates) {
                 val currentPrefix = if (dir.isEmpty()) "" else "$dir/"
                 val testAssetPath = currentPrefix + "text_encoder.onnx"
                 
                 // Check if asset exists
                 val assetExists = try {
                     context.assets.list(dir)?.contains("text_encoder.onnx") == true
                 } catch (e: Exception) { false }
                 
                 if (assetExists) {
                     Log.d("Antigravity-SD", "Checking candidate: '$testAssetPath'...")
                     
                     // Copy to TEMP file for validation
                     val tempFile = java.io.File(context.filesDir, "temp_text_encoder_check.onnx")
                     try {
                         context.assets.open(testAssetPath).use { input ->
                             java.io.FileOutputStream(tempFile).use { output -> input.copyTo(output) }
                         }
                         
                         // Validate Session
                         val session = env.createSession(tempFile.absolutePath, npuPropeties)
                         val inputs = session.inputNames
                         val outputs = session.outputNames
                         session.close()
                         
                         if (inputs.contains("input_ids")) {
                             Log.d("Antigravity-SD", "VALID Match Found in '$dir'! Inputs: $inputs")
                             assetPrefix = currentPrefix
                             
                             // Rename temp to real
                             val realFile = java.io.File(context.filesDir, "text_encoder.onnx")
                             if (realFile.exists()) realFile.delete()
                             tempFile.renameTo(realFile)
                             
                             foundValidTextEncoder = true
                             break
                         } else {
                             Log.w("Antigravity-SD", "INVALID Candidate in '$dir'. Found inputs: $inputs. Skipping.")
                             tempFile.delete() // Cleanup bad file
                         }
                     } catch (e: Exception) {
                         Log.w("Antigravity-SD", "Error validating candidate '$dir': ${e.message}")
                         tempFile.delete()
                     }
                 }
            }
            
            if (!foundValidTextEncoder) {
                // Critical Failure
                val rootList = context.assets.list("")?.joinToString()
                Log.e("Antigravity-SD", "CRITICAL: No valid 'text_encoder.onnx' found in any known path! Root: [$rootList]")
                throw java.io.FileNotFoundException("Could not find a valid text_encoder.onnx (with input='input_ids') in assets.")
            }

            // 3. Install Remaining Assets (using verified prefix)
            copyAssetToFile(assetPrefix + "vae_decoder.onnx")
            copyAssetToFile(assetPrefix + "unet_fp16.onnx")
            copyAssetToFile(assetPrefix + "weights.pb")

            onStatus("Loading Models...")
            Log.d("Antigravity-SD", "Loading Models from ${context.filesDir.absolutePath}...")
            
            val filesDir = context.filesDir
            
            // Text Encoder (Already installed & validated)
            val textEncoderPath = java.io.File(filesDir, "text_encoder.onnx").absolutePath
            textEncoderSession = env.createSession(textEncoderPath, npuPropeties)
            
            // VALIDATION: Check if we loaded the wrong model (e.g. UNet instead of TextEncoder)
            // The error "unknown input name input_ids, expected one of [sample]" means we have UNet loaded here.
            // VALIDATION: Check if we loaded the wrong model (e.g. UNet instead of TextEncoder)
            val inputs = textEncoderSession!!.inputNames
            val outputs = textEncoderSession!!.outputNames
            
            if (!inputs.contains("input_ids")) {
                 val corruptedFile = java.io.File(textEncoderPath)
                 val fileSize = if (corruptedFile.exists()) corruptedFile.length() else -1L
                 
                 Log.e("Antigravity-SD", "CRITICAL MODEL ERROR: Text Encoder Invalid")
                 Log.e("Antigravity-SD", "  - Path: $textEncoderPath")
                 Log.e("Antigravity-SD", "  - Size: ${fileSize / 1024 / 1024} MB")
                 Log.e("Antigravity-SD", "  - Inputs Found: $inputs")
                 Log.e("Antigravity-SD", "  - Outputs Found: $outputs")
                 
                 if (filesDir.list()?.contains("unet_fp16.onnx") == true) {
                     val unetSize = java.io.File(filesDir, "unet_fp16.onnx").length()
                     if (fileSize == unetSize) {
                         Log.e("Antigravity-SD", "  - DIAGNOSIS: File size matches UNet! You likely copied UNet to text_encoder.onnx.")
                     }
                 }
                 
                 Log.w("Antigravity-SD", "Attempting self-healing (Delete & Re-Copy)...")
                 
                 textEncoderSession!!.close()
                 if (corruptedFile.exists()) corruptedFile.delete()
                 
                 // Force Re-copy from the correct autodetected prefix
                 copyAssetToFile(assetPrefix + "text_encoder.onnx", overwrite = true)
                 
                 // Reload
                 textEncoderSession = env.createSession(textEncoderPath, npuPropeties)
                 
                 if (!textEncoderSession!!.inputNames.contains("input_ids")) {
                     val newInputs = textEncoderSession!!.inputNames
                     Log.e("Antigravity-SD", "RE-VALIDATION FAILED: Inputs are still $newInputs")
                     throw RuntimeException("Text Encoder Invalid. See Logcat Antigravity-SD for details.")
                 }
                 Log.i("Antigravity-SD", "Self-healing successful. Text Encoder matches expected signature.")
            }
            
            // UNet FP16 (NPU) - this will automatically load weights.pb from the same dir
            val unetFile = java.io.File(filesDir, "unet_fp16.onnx")
            if (!unetFile.exists()) throw java.io.FileNotFoundException("unet_fp16.onnx not found!")
            
            unetSession = env.createSession(unetFile.absolutePath, npuPropeties)
            
            // VAE Decoder (CPU - usually safer/fast enough, but could try NPU if model supports it)
            val vaePath = java.io.File(filesDir, "vae_decoder.onnx").absolutePath
            vaeDecoderSession = env.createSession(vaePath, cpuOptions)
            
            Log.i("Antigravity-SD", "SD Engine Ready. Device: $activeDevice")
            onStatus("Ready ($activeDevice)")

        } catch (e: Exception) {
            Log.e("Antigravity-SD", "Initialization failed", e)
            onStatus("Error: ${e.message}")
            throw e
        }
    }

    // ... [Helpers copyAllModelsFromAssets, getOrMergeWeights kept as is by diff context usually, but here I am targeting lines 57-138 mostly for init]
    // Wait, I need to verify I am not wiping helpers if I only target init.
    // The previous diff ended at line 138 (end of init).
    // I will replace generateImage (line 197+) separately or in same block if I can bridge it. 
    // It's safer to do 2 chunks or just target the functions individually.
    // I will target `initializeModels` first (S25 Threads).
    // Actually, I can combine if I use MultiReplace.
    // Let's stick to replacing `initializeModels` block (lines 57-138) and `generateImage` block (197-285).
    // But this tool call only allows single contiguous or I use multi_replace.
    // I will use `replace_file_content` for `initializeModels` now, then `generateImage`.
    
    // Ah, wait. I will do ONE BIG replacement for the whole logic part if feasible, or just iterate.
    // Let's do `initializeModels` optimization first.
    
    // NO, I will use `multi_replace_file_content` to do both at once.
    // ...
    // Actually, let's just do `initializeModels` modification first manually here.
    
    // Changing strategy: I will replace the whole file content for lines 99-102 (Threads) and 197-285 (Generate).
    // Let's use `multi_replace_file_content`.



    private fun copyAssetToFile(assetPath: String, overwrite: Boolean = false) {
        val destFilename = java.io.File(assetPath).name
        val file = java.io.File(context.filesDir, destFilename)
        
        if (!overwrite && file.exists() && file.length() > 0) {
            Log.d("Antigravity-SD", "$destFilename already exists (${file.length()} bytes). Skipping copy.")
            return
        }
        
        Log.d("Antigravity-SD", "Installing $assetPath as $destFilename (Overwrite: $overwrite)...")
        try {
            context.assets.open(assetPath).use { input ->
                java.io.FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
            Log.d("Antigravity-SD", "Installed $destFilename (${file.length()} bytes).")
        } catch (e: Exception) {
            Log.e("Antigravity-SD", "Failed to copy $assetPath", e)
            throw e
        }
    }

    fun generateImage(prompt: String): Flow<GenerationStatus> = flow {
        val startTime = System.currentTimeMillis()
        val configMsg = "Device: $activeDevice | Steps: $numInferenceSteps | CFG: 7.5 | Scheduler: Euler Ancestral (S25 Opt) | Threading: Default"
        emit(GenerationStatus.Started(configMsg))
        emit(GenerationStatus.InProgress(0, "Encoding Text..."))

        // 1. Text Embeddings
        val tokenIds = tokenizer.tokenize(prompt) 
        val condEmbeddings = encodeText(tokenIds)
        val uncondEmbeddings = encodeText(IntArray(77) { 49407 }) // Empty prompt

        // Concatenate for Batch Size 2 (Uncond, Cond)
        val textEmbeddings = FloatArray(2 * 77 * 768)
        System.arraycopy(uncondEmbeddings, 0, textEmbeddings, 0, 77 * 768)
        System.arraycopy(condEmbeddings, 0, textEmbeddings, 77 * 768, 77 * 768)
        
        // 2. Scheduler Setup
        val timesteps = getTimesteps(numInferenceSteps)
        val initSigma = sigmas[timesteps[0].toInt()]
        var latents = generateRandomLatents(1, 4, 64, 64)
        
        // Scale initial latents
        for (i in latents.indices) {
            latents[i] = latents[i] * sqrt(initSigma.pow(2) + 1)
        }

        emit(GenerationStatus.InProgress(10, "Starting Diffusion (Euler A S25)..."))

        // PRE-ALLOCATED BUFFERS
        val chunkSize = 4 * 64 * 64
        val latentInput = FloatArray(chunkSize)
        val doubleLatentsBuffer = FloatBuffer.allocate(2 * chunkSize)
        val guidance = 7.5f
        val rand = java.util.Random()
        
        // Hoist Buffer Wrap (Optimization: 1 wrap vs 20)
        val stepCombinedEmbeddings = FloatBuffer.wrap(textEmbeddings)

        // 4. Denoising Loop
        for ((index, t) in timesteps.withIndex()) {
            val stepStart = System.nanoTime()
            
            // Scheduler Step: Euler Ancestral Math
            val sigma = sigmas[t.toInt()]
             
            // Calculate Sigma Up/Down/To for Ancestral Sampling
            val sigmaTo = if (index + 1 < timesteps.size) sigmas[timesteps[index+1].toInt()] else 0.0f
            
            // Sigma Up (Ancestral Noise Magnitude)
            // sigma_up = sqrt(sigma_to^2 * (sigma_from^2 - sigma_to^2) / sigma_from^2)
            val sigmaUp = sqrt((sigmaTo.pow(2) * (sigma.pow(2) - sigmaTo.pow(2))) / sigma.pow(2))
            
            // Sigma Down (Resulting Sigma after step)
            // sigma_down = sqrt(sigma_to^2 - sigma_up^2)
            val sigmaDown = sqrt(sigmaTo.pow(2) - sigmaUp.pow(2))
            
            // dt = sigma_down - sigma_from
            val dt = sigmaDown - sigma
            
            // Scale Input
            val scale = 1.0f / sqrt(sigma.pow(2) + 1)
            for (i in 0 until chunkSize) {
                latentInput[i] = latents[i] * scale
            }
            
            // Batch Expansion
            doubleLatentsBuffer.clear()
            doubleLatentsBuffer.put(latentInput)
            doubleLatentsBuffer.put(latentInput)
            doubleLatentsBuffer.flip()

            // UNet Inference
            val unetStart = System.nanoTime()
            // Reset position of reused buffer
            stepCombinedEmbeddings.rewind() 
            val noisePred = runUNet(doubleLatentsBuffer, t, stepCombinedEmbeddings)
            val unetTime = (System.nanoTime() - unetStart) / 1_000_000.0
            
            // Fused Euler A Update: Pred + Noise
            for (i in 0 until chunkSize) {
                // CFG
                val uncond = noisePred[i]
                val cond = noisePred[i + chunkSize]
                val noise = uncond + guidance * (cond - uncond)
                
                // Euler Step: x = x + d * dt
                // d = noise (in this formulation, noisePred is essentially the derivative)
                latents[i] += noise * dt
                
                // Ancestral Noise Addition (if sigmaUp > 0)
                if (sigmaTo > 0.0f) { // Not last step
                     latents[i] += rand.nextGaussian().toFloat() * sigmaUp
                }
            }
            
            val stepTotalTime = (System.nanoTime() - stepStart) / 1_000_000.0
            if (index == 0 || index == numInferenceSteps -1 ) {
                 Log.d("Antigravity-SD", "Step ${index+1}: Total=${String.format("%.2f", stepTotalTime)}ms | UNet=${String.format("%.2f", unetTime)}ms")
            }
            
            val progress = 10 + ((index + 1).toFloat() / numInferenceSteps * 80).roundToInt()
            emit(GenerationStatus.InProgress(progress, "Step ${index + 1}/$numInferenceSteps (${String.format("%.0f", stepTotalTime)}ms)"))
        }

        // 5. VAE Decode
        emit(GenerationStatus.InProgress(95, "Decoding..."))
        
        // VAE Double Scaling Fix:
        // Removed the pre-scaling loop here. Passing latents directly.
        // runVAEDecoder handles formatting.
        
        val image = runVAEDecoder(latents) // Sent directly
        val endTime = System.currentTimeMillis()
        emit(GenerationStatus.Success(image, endTime - startTime))
    }.flowOn(Dispatchers.Default) // CRITICAL: Move entire pipeline off main thread

    private fun getTimesteps(steps: Int): IntArray {
        // Linear spacing 999 -> 0
        val ts = IntArray(steps)
        val stepSize = 1000 / steps
        for (i in 0 until steps) {
            ts[i] = 999 - (i * stepSize)
        }
        return ts
    }

    private suspend fun encodeText(tokenIds: IntArray): FloatArray {
        if (textEncoderSession == null) return FloatArray(77 * 768) // Mock
        return withContext(Dispatchers.Default) {
            val inputTensor = OnnxTensor.createTensor(env, IntBuffer.wrap(tokenIds), longArrayOf(1, 77))
            val result = textEncoderSession!!.run(mapOf("input_ids" to inputTensor))
            val output = result[0] as OnnxTensor
            val floatBuffer = output.floatBuffer
            val out = FloatArray(floatBuffer.remaining())
            floatBuffer.get(out)
            result.close()
            out
        }
    }

    private fun toFP16Buffer(floats: FloatBuffer): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(floats.remaining() * 2)
        buffer.order(ByteOrder.nativeOrder())
        while (floats.hasRemaining()) {
            buffer.putShort(Half.toHalf(floats.get()))
        }
        buffer.rewind()
        floats.rewind() // Reset source buffer position
        return buffer
    }

    private fun fromFP16ToFloatArray(buffer: ByteBuffer): FloatArray {
        buffer.order(ByteOrder.nativeOrder())
        val floatCount = buffer.remaining() / 2
        val out = FloatArray(floatCount)
        for (i in 0 until floatCount) {
             val s = buffer.short
             out[i] = Half.toFloat(s)
        }
        return out
    }

    private suspend fun runUNet(sample: FloatBuffer, timestep: Int, encoderHiddenStates: FloatBuffer): FloatArray {
         if (unetSession == null) return FloatArray(2 * 4 * 64 * 64) // Mock
         return withContext(Dispatchers.Default) {
             var sampleTensor: OnnxTensor? = null
             var tTensor: OnnxTensor? = null
             var encoderTensor: OnnxTensor? = null
             var result: ai.onnxruntime.OrtSession.Result? = null

             try {
                 // 1. Prepare Inputs (Convert to FP16)
                 val sampleFP16 = toFP16Buffer(sample)
                 val encoderFP16 = toFP16Buffer(encoderHiddenStates)
                 val tFP16 = toFP16Buffer(FloatBuffer.wrap(floatArrayOf(timestep.toFloat())))

                 // 2. Create Tensors (FP16 Type)
                 sampleTensor = OnnxTensor.createTensor(env, sampleFP16, longArrayOf(2, 4, 64, 64), OnnxJavaType.FLOAT16)
                 tTensor = OnnxTensor.createTensor(env, tFP16, longArrayOf(1), OnnxJavaType.FLOAT16)
                 encoderTensor = OnnxTensor.createTensor(env, encoderFP16, longArrayOf(2, 77, 768), OnnxJavaType.FLOAT16)

                 val inputs = mapOf(
                     "sample" to sampleTensor,
                     "timestep" to tTensor,
                     "encoder_hidden_states" to encoderTensor
                 )

                 // 3. Run Inference
                 result = unetSession!!.run(inputs)

                 // 4. Handle Output
                 val outputTensor = result!!.get(0) as OnnxTensor

                 if (outputTensor.info.type == OnnxJavaType.FLOAT16) {
                     val fb = outputTensor.byteBuffer
                     fromFP16ToFloatArray(fb)
                 } else {
                     val fb = outputTensor.floatBuffer
                     val out = FloatArray(fb.remaining())
                     fb.get(out)
                     out
                 }
             } finally {
                 // CRITICAL: Close native resources to prevent Binder Transaction Failure / Leak
                 result?.close()
                 sampleTensor?.close()
                 tTensor?.close()
                 encoderTensor?.close()
             }
         }
    }

    private suspend fun runVAEDecoder(latents: FloatArray): Bitmap {
        if (vaeDecoderSession == null) return createNoiseBitmap(512, 512)
        return withContext(Dispatchers.Default) {
            val scaledLatents = FloatArray(latents.size) { i -> latents[i] / 0.18215f }
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(scaledLatents), longArrayOf(1, 4, 64, 64))

             val result = vaeDecoderSession!!.run(mapOf("latent_sample" to inputTensor))
             val output = result[0] as OnnxTensor // [1, 3, 512, 512]
             val fb = output.floatBuffer

             // Convert float [0, 1] or [-1, 1] to Colors
             // VAE output is usually [-1, 1]
             // Convert NCHW [1, 3, 512, 512] to ARGB
             val width = 512
             val height = 512
             val colors = IntArray(width * height)

             // Plane offsets for NCHW layout
             val rOffset = 0
             val gOffset = width * height
             val bOffset = 2 * width * height

             for (y in 0 until height) {
                 for (x in 0 until width) {
                     val i = y * width + x

                     val rVal = fb[rOffset + i]
                     val gVal = fb[gOffset + i]
                     val bVal = fb[bOffset + i]

                     // Helper strict denormalization: (val + 1) * 127.5
                     val r = ((rVal + 1.0f) * 127.5f).coerceIn(0f, 255f).toInt()
                     val g = ((gVal + 1.0f) * 127.5f).coerceIn(0f, 255f).toInt()
                     val b = ((bVal + 1.0f) * 127.5f).coerceIn(0f, 255f).toInt()

                     // Alpha is always 255 (0xFF)
                     colors[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                 }
             }

             result.close()
             inputTensor.close()
             Bitmap.createBitmap(colors, width, height, Bitmap.Config.ARGB_8888)
        }
    }
    
    private fun generateRandomLatents(b: Int, c: Int, h: Int, w: Int): FloatArray {
         val rand = java.util.Random()
         return FloatArray(b * c * h * w) { rand.nextGaussian().toFloat() }
    }
    
    // ... [Status classes and Mock Bitmap Helper] ...
    private fun createNoiseBitmap(w: Int, h: Int): Bitmap {
        val conf = Bitmap.Config.ARGB_8888
        val bmp = Bitmap.createBitmap(w, h, conf)
        for(x in 0 until w step 10) {
            for (y in 0 until h step 10) {
                 bmp.setPixel(x, y, Color.rgb((0..255).random(), (0..255).random(), (0..255).random()))
            }
        }
        return bmp
    }

    sealed class GenerationStatus {
        data class InProgress(val progress: Int, val message: String) : GenerationStatus()
        data class Started(val config: String) : GenerationStatus()
        class Success(val image: Bitmap, val timeMs: Long) : GenerationStatus()
        class Error(val exception: Throwable) : GenerationStatus()
    }
}
