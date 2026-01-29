package com.antigravity.sd

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.antigravity.sd.logic.StableDiffusionManager
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var sdManager: StableDiffusionManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        sdManager = StableDiffusionManager(this)

        setContent {
            AntigravitySDTheme {
                MainScreen(sdManager)
            }
        }
    }
}

@Composable
fun AntigravitySDTheme(content: @Composable () -> Unit) {
    // Premium Dark Theme Vibe
    val DarkColors = darkColorScheme(
        primary = Color(0xFFD0BCFF),
        secondary = Color(0xFFCCC2DC),
        tertiary = Color(0xFFEFB8C8),
        background = Color(0xFF121212),
        surface = Color(0xFF1E1E1E),
        onPrimary = Color(0xFF381E72),
        onSecondary = Color(0xFF332D41),
        onTertiary = Color(0xFF492532),
        onBackground = Color(0xFFE6E1E5),
        onSurface = Color(0xFFE6E1E5),
    )

    MaterialTheme(
        colorScheme = DarkColors,
        typography = Typography(),
        content = content
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(sdManager: StableDiffusionManager) {
    var prompt by remember { mutableStateOf("") }
    var generatedImage by remember { mutableStateOf<Bitmap?>(null) }
    var progress by remember { mutableFloatStateOf(0f) }
    var statusMessage by remember { mutableStateOf("Ready to Create") }
    var generationStats by remember { mutableStateOf("") } // New State for Info
    var isGenerating by remember { mutableStateOf(false) }
    var generationJob by remember { mutableStateOf<kotlinx.coroutines.Job?>(null) }

    val scope = rememberCoroutineScope()
    val context = LocalContext.current
    val focusManager = LocalFocusManager.current
    val scrollState = rememberScrollState() // Scroll State

    // Initialize Models nicely
    LaunchedEffect(Unit) {
        try {
            sdManager.initializeModels { status ->
                statusMessage = status
                if (status.contains("Merging")) {
                    progress = 1f 
                } else if (status.contains("Ready")) {
                    progress = 0f
                }
            }
        } catch (e: Exception) {
            statusMessage = "Model Load Failed: ${e.message}"
        }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize().imePadding(), // Smart Keyboard handling
        containerColor = MaterialTheme.colorScheme.background
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp)
                .verticalScroll(scrollState), // Enable Scroll
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            // Header
            Text(
                text = "Antigravity-SD",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
                letterSpacing = 1.sp
            )
            
            Text(
                text = "Powered by S25 NPU",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.secondary.copy(alpha = 0.7f)
            )

            Spacer(modifier = Modifier.height(10.dp))

            // Image Area
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(16.dp))
                    .background(MaterialTheme.colorScheme.surface),
                contentAlignment = Alignment.Center
            ) {
                if (generatedImage != null) {
                    Image(
                        bitmap = generatedImage!!.asImageBitmap(),
                        contentDescription = "Generated Image",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    if (isGenerating) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(48.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                    } else {
                        Text(
                            text = "Enter a prompt and dream.",
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
                        )
                    }
                }
            }

            // Stats Information Display
            if (generationStats.isNotEmpty()) {
                Text(
                    text = generationStats,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.tertiary,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 8.dp)
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
                        .padding(8.dp)
                )
            }

            // Progress Bar (Animated)
            val animatedProgress by animateFloatAsState(targetValue = progress / 100f, label = "Progress")
            
            Column(modifier = Modifier.fillMaxWidth()) {
                if (isGenerating || progress > 0) {
                    LinearProgressIndicator(
                        progress = animatedProgress,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(RoundedCornerShape(4.dp)),
                        trackColor = MaterialTheme.colorScheme.surface,
                        color = MaterialTheme.colorScheme.tertiary
                    )
                }
                
                // Always visible Status Message
                Text(
                    text = statusMessage,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 8.dp),
                    color = if (statusMessage.contains("Error") || statusMessage.contains("Failed")) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            // Input Area
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                label = { Text("Imagine anything...") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = false,
                maxLines = 3,
                enabled = !isGenerating,
                shape = RoundedCornerShape(12.dp),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { focusManager.clearFocus() })
            )

            // Button with Stop Functionality
            Button(
                onClick = {
                    Log.d("Antigravity-SD", "Generate Button Clicked. isGenerating=$isGenerating")
                    if (isGenerating) {
                        // Cancel Logic
                        generationJob?.cancel()
                        isGenerating = false
                        statusMessage = "Cancelled"
                        progress = 0f
                        generationStats = ""
                    } else {
                        // Generate Logic
                        if (prompt.isBlank()) {
                            Toast.makeText(context, "Please enter a prompt!", Toast.LENGTH_SHORT).show()
                            return@Button
                        }
                        focusManager.clearFocus()
                        isGenerating = true
                        progress = 0f
                        generatedImage = null
                        generationStats = "" // Reset stats
                        
                        generationJob = scope.launch {
                            try {
                                sdManager.generateImage(prompt).collect { status ->
                                    if (!kotlinx.coroutines.currentCoroutineContext().isActive) return@collect // Safety check
                                    when (status) {
                                        is StableDiffusionManager.GenerationStatus.Started -> {
                                            generationStats = status.config
                                        }
                                        is StableDiffusionManager.GenerationStatus.InProgress -> {
                                            progress = status.progress.toFloat()
                                            statusMessage = status.message
                                        }
                                        is StableDiffusionManager.GenerationStatus.Success -> {
                                            generatedImage = status.image
                                            isGenerating = false
                                            progress = 100f
                                            statusMessage = "Completed"
                                            val timeSec = String.format("%.2f", status.timeMs / 1000f)
                                            generationStats += "\nDone in ${timeSec}s"
                                        }
                                        is StableDiffusionManager.GenerationStatus.Error -> {
                                            isGenerating = false
                                            statusMessage = "Error: ${status.exception.localizedMessage}"
                                            Toast.makeText(context, "Error: ${status.exception.message}", Toast.LENGTH_LONG).show()
                                            status.exception.printStackTrace()
                                        }
                                    }
                                }
                            } catch (e: kotlinx.coroutines.CancellationException) {
                                isGenerating = false
                                statusMessage = "Cancelled"
                                progress = 0f
                            } catch (e: Exception) {
                                isGenerating = false
                                statusMessage = "Fatal Error: ${e.message}"
                                Log.e("Antigravity-SD", "UNHANDLED EXCEPTION IN GENERATION JOB", e)
                                e.printStackTrace()
                            }
                        }
                    }
                },

                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                enabled = true, // Always enabled so we can Stop
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isGenerating) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                    contentColor = if (isGenerating) MaterialTheme.colorScheme.onError else MaterialTheme.colorScheme.onPrimary
                )
            ) {
                Text(
                    text = if (isGenerating) "Stop Generation" else "Generate",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}
