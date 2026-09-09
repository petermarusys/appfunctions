/*
 * Copyright 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.appfunctions.agent.data

import android.content.Context
import android.util.Log
import androidx.appfunctions.metadata.AppFunctionMetadata
import com.example.appfunctions.agent.domain.LlmInput
import com.example.appfunctions.agent.domain.LlmProvider
import com.example.appfunctions.agent.domain.LlmResponse
import com.example.appfunctions.agent.domain.LlmResponsePart
import com.example.appfunctions.agent.domain.ToolOutput
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.Role
import com.google.ai.edge.litertlm.tool
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.time.LocalDate
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Implementation of [LlmProvider] for Gemma 4 (E4B / E2B) edge and on-device LLM execution.
 *
 * Backed by Google LiteRT-LM (litertlm-android) with CPU/NPU acceleration,
 * tool calling support via OpenApiTool declarations, and graceful fallback to
 * rule-based intent parsing when the on-device model file is unavailable.
 */
@Singleton
class GemmaProviderImpl private constructor(
    private val toolConverter: GeminiToolConverter,
    private val context: Context?,
    @Suppress("UNUSED_PARAMETER") isTest: Boolean,
) : LlmProvider {

    @Inject
    constructor(
        toolConverter: GeminiToolConverter,
        @ApplicationContext context: Context,
    ) : this(toolConverter, context as Context?, false)

    internal constructor(toolConverter: GeminiToolConverter) : this(toolConverter, null, true)

    private val initMutex = Mutex()
    private var engine: Engine? = null
    private var engineInitAttempted = false

    private val conversations = ConcurrentHashMap<String, Conversation>()
    private val sessionHistory = ConcurrentHashMap<String, MutableList<GemmaTurn>>()

    override suspend fun generateResponse(
        previousInteractionId: String?,
        input: LlmInput,
        tools: List<AppFunctionMetadata>,
        apiKey: String,
        modelName: String,
        serviceTier: ServiceTier,
    ): LlmResponse {
        val newInteractionId = UUID.randomUUID().toString()

        val history =
            if (previousInteractionId != null && sessionHistory.containsKey(previousInteractionId)) {
                sessionHistory[previousInteractionId]?.toMutableList() ?: mutableListOf()
            } else {
                mutableListOf()
            }

        when (input) {
            is LlmInput.UserMessage -> history.add(GemmaTurn.User(input.text))
            is LlmInput.ToolResponse -> history.add(GemmaTurn.ToolResult(input.outputs))
        }

        // 1. Attempt on-device LiteRT-LM inference if the model is available
        val activeEngine = getOrInitEngine(modelName)
        if (activeEngine != null) {
            try {
                val responseParts = withContext(Dispatchers.IO) {
                    executeLiteRtLmInference(
                        engine = activeEngine,
                        previousInteractionId = previousInteractionId,
                        newInteractionId = newInteractionId,
                        input = input,
                        tools = tools,
                    )
                }
                if (responseParts != null && responseParts.isNotEmpty()) {
                    history.add(GemmaTurn.Assistant(responseParts))
                    sessionHistory[newInteractionId] = history
                    return LlmResponse.Success(
                        interactionId = newInteractionId,
                        parts = responseParts,
                    )
                }
            } catch (e: Exception) {
                Log.w(TAG, "LiteRT-LM inference encountered an issue, falling back: ${e.message}", e)
            }
        }

        // 2. Fallback heuristic inference (supports unit tests and uninitialized/missing model file)
        val responseParts = processTurnFallback(history, tools, modelName)
        history.add(GemmaTurn.Assistant(responseParts))
        sessionHistory[newInteractionId] = history

        return LlmResponse.Success(
            interactionId = newInteractionId,
            parts = responseParts,
        )
    }

    private suspend fun getOrInitEngine(modelName: String): Engine? {
        if (context == null) return null
        return initMutex.withLock {
            if (engine != null) return@withLock engine
            if (engineInitAttempted) return@withLock null
            engineInitAttempted = true

            val nativeLibDir = context.applicationInfo.nativeLibraryDir
            val cacheDir =
                context.getExternalFilesDir(null)?.absolutePath ?: context.cacheDir.absolutePath

            // 1. Try Qualcomm NPU (QNN) backend with SM8750 model if available
            val qnnModelPath = findQnnModelPath()
            if (qnnModelPath != null) {
                try {
                    Log.i(TAG, "Attempting LiteRT-LM Engine initialization with Qualcomm NPU at $qnnModelPath (nativeLibDir=$nativeLibDir)...")
                    val qnnConfig = EngineConfig(
                        modelPath = qnnModelPath,
                        backend = Backend.NPU(nativeLibraryDir = nativeLibDir),
                        cacheDir = cacheDir,
                    )
                    val qnnEngine = Engine(qnnConfig)
                    qnnEngine.initialize()
                    Log.i(TAG, "LiteRT-LM Engine successfully initialized with Qualcomm NPU backend.")
                    engine = qnnEngine
                    return@withLock qnnEngine
                } catch (e: Throwable) {
                    Log.w(TAG, "Failed to initialize Qualcomm NPU backend: ${e.message}. Falling back to CPU backend.", e)
                }
            }

            // 2. Fall back to CPU backend
            val cpuModelPath = findCpuModelPath(modelName)
            if (cpuModelPath == null) {
                Log.w(TAG, "Gemma model file not found for $modelName. Falling back to heuristic provider.")
                return@withLock null
            }

            try {
                Log.i(TAG, "Initializing LiteRT-LM Engine with CPU backend at $cpuModelPath...")
                val config = EngineConfig(
                    modelPath = cpuModelPath,
                    backend = Backend.CPU(),
                    maxNumTokens = 8192,
                    cacheDir = cacheDir,
                )
                val newEngine = Engine(config)
                newEngine.initialize()
                Log.i(TAG, "LiteRT-LM Engine successfully initialized with CPU backend.")
                engine = newEngine
                newEngine
            } catch (e: Throwable) {
                Log.e(TAG, "Failed to initialize LiteRT-LM Engine: ${e.message}", e)
                null
            }
        }
    }

    private fun findQnnModelPath(): String? {
        val candidates = listOf(
            "/data/local/tmp/gemma-4-E2B-it_qualcomm_sm8750.litertlm",
            context?.let { File(it.filesDir, "gemma-4-E2B-it_qualcomm_sm8750.litertlm").absolutePath },
            context?.let { File(it.getExternalFilesDir(null), "gemma-4-E2B-it_qualcomm_sm8750.litertlm").absolutePath },
        ).filterNotNull()
        return candidates.firstOrNull { File(it).exists() }
    }

    private fun findCpuModelPath(modelName: String): String? {
        val candidates = listOf(
            "/data/local/tmp/gemma-4-E2B-it.litertlm",
            "/data/local/tmp/gemma-4-E4B-it.litertlm",
            context?.let { File(it.filesDir, "gemma-4-E2B-it.litertlm").absolutePath },
            context?.let { File(it.getExternalFilesDir(null), "gemma-4-E2B-it.litertlm").absolutePath },
            "/data/local/tmp/model.litertlm",
        ).filterNotNull()
        return candidates.firstOrNull { File(it).exists() }
    }

    private fun executeLiteRtLmInference(
        engine: Engine,
        previousInteractionId: String?,
        newInteractionId: String,
        input: LlmInput,
        tools: List<AppFunctionMetadata>,
    ): List<LlmResponsePart>? {
        val userQuery = when (input) {
            is LlmInput.UserMessage -> input.text
            is LlmInput.ToolResponse -> ""
        }
        val relevantTools = filterRelevantTools(tools, userQuery)
        val convertedTools = relevantTools
            .sortedByDescending { it.id.startsWith(it.packageName) }
            .distinctBy { toolConverter.getToolName(it) }

        val toolMap = mutableMapOf<String, AppFunctionMetadata>()
        for (tool in convertedTools) {
            toolMap[toolConverter.getToolName(tool)] = tool
            toolMap[tool.id] = tool
        }

        val conversation: Conversation
        if (previousInteractionId != null && conversations.containsKey(previousInteractionId)) {
            conversation = conversations[previousInteractionId]!!
        } else {
            conversations.values.forEach { runCatching { it.close() } }
            conversations.clear()

            val toolProviders = convertedTools.mapNotNull { toolMetadata ->
                try {
                    val toolJson = toolConverter.convert(toolMetadata)
                    val openApiTool = object : OpenApiTool {
                        override fun getToolDescriptionJsonString(): String = toolJson.toString()
                        override fun execute(paramsJsonString: String): String = ""
                    }
                    tool(openApiTool)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to create OpenApiTool for ${toolMetadata.id}: ${e.message}")
                    null
                }
            }

            val config = ConversationConfig(
                systemInstruction = Contents.of(getSystemInstruction()),
                tools = toolProviders,
                automaticToolCalling = false,
            )
            conversation = engine.createConversation(config)
        }

        val responseMessage: Message = when (input) {
            is LlmInput.UserMessage -> {
                Log.d(TAG, "LiteRT-LM sending UserMessage: ${input.text}")
                conversation.sendMessage(input.text)
            }
            is LlmInput.ToolResponse -> {
                Log.d(TAG, "LiteRT-LM sending ToolResponse: ${input.outputs.size} outputs")
                val toolContents = input.outputs.map { output ->
                    val toolName = toolMap.entries.firstOrNull { it.value.id == output.functionId }?.key
                        ?: output.functionId
                    Content.ToolResponse(toolName, output.result)
                }
                val message = Message.tool(Contents.of(toolContents))
                conversation.sendMessage(message)
            }
        }

        conversations[newInteractionId] = conversation
        return parseLiteRtLmMessage(responseMessage, tools, toolMap)
    }

    private fun parseLiteRtLmMessage(
        message: Message,
        tools: List<AppFunctionMetadata>,
        toolMap: Map<String, AppFunctionMetadata>,
    ): List<LlmResponsePart> {
        val parts = mutableListOf<LlmResponsePart>()

        val toolCalls = message.toolCalls
        if (toolCalls.isNotEmpty()) {
            for (toolCall in toolCalls) {
                val matchedTool = matchTool(toolCall.name, tools)

                if (matchedTool != null) {
                    val normalizedArgs = normalizeToolArguments(matchedTool, toolCall.arguments)
                    Log.i(TAG, "Successfully matched native ToolCall: ${matchedTool.id} args=$normalizedArgs")
                    parts.add(
                        LlmResponsePart.ToolCall(
                            packageName = matchedTool.packageName,
                            functionId = matchedTool.id,
                            arguments = normalizedArgs,
                            callId = UUID.randomUUID().toString(),
                        )
                    )
                } else {
                    Log.w(TAG, "Unrecognized toolCall name: ${toolCall.name}")
                }
            }
        }

        val textContents = message.contents.contents.filterIsInstance<Content.Text>().map { it.text }
        val combinedText = textContents.joinToString("\n").trim()

        if (parts.isNotEmpty()) {
            if (combinedText.isNotBlank()) {
                parts.add(0, LlmResponsePart.Text(combinedText))
            }
            return parts
        }

        if (combinedText.isNotBlank()) {
            val textToolCall = parseToolCallFromText(combinedText, tools)
            if (textToolCall != null) {
                return listOf(textToolCall)
            }
            return listOf(LlmResponsePart.Text(combinedText))
        }

        return listOf(LlmResponsePart.Text(""))
    }

    private fun matchTool(
        toolCallName: String,
        tools: List<AppFunctionMetadata>,
    ): AppFunctionMetadata? {
        // 1. Exact match with toolConverter name
        tools.firstOrNull { toolConverter.getToolName(it).equals(toolCallName, ignoreCase = true) }
            ?.let { return it }

        // 2. Exact match with function ID
        tools.firstOrNull { it.id.equals(toolCallName, ignoreCase = true) }
            ?.let { return it }

        // 3. Match normalized name (strip special characters)
        val cleanCallName = toolCallName.replace(Regex("[^a-zA-Z0-9]"), "").lowercase()
        tools.firstOrNull { tool ->
            val cleanId = tool.id.replace(Regex("[^a-zA-Z0-9]"), "").lowercase()
            val cleanToolName = toolConverter.getToolName(tool).replace(Regex("[^a-zA-Z0-9]"), "").lowercase()
            cleanId.endsWith(cleanCallName) || cleanCallName.endsWith(cleanId) ||
                cleanToolName.endsWith(cleanCallName) || cleanCallName.endsWith(cleanToolName)
        }?.let { return it }

        // 4. Match simple function name suffix (e.g. "makeCall", "searchContacts", "sendMessage")
        val simpleFunctionName = toolCallName.substringAfterLast("_").lowercase()
        tools.firstOrNull { tool ->
            val toolSimple = tool.id.substringAfterLast("#").substringAfterLast(".").lowercase()
            toolSimple == simpleFunctionName ||
                toolSimple.endsWith(simpleFunctionName) ||
                simpleFunctionName.endsWith(toolSimple) ||
                (simpleFunctionName == "call" && toolSimple.contains("call")) ||
                (simpleFunctionName == "message" && toolSimple.contains("message"))
        }?.let { return it }

        // 5. Keyword search in tool id
        val keywords = listOf("call", "contact", "message", "email", "search")
        for (keyword in keywords) {
            if (toolCallName.lowercase().contains(keyword)) {
                tools.firstOrNull { it.id.lowercase().contains(keyword) }?.let { return it }
            }
        }
        return null
    }

    private fun normalizeToolArguments(
        tool: AppFunctionMetadata,
        rawArgs: Map<String, Any?>,
    ): Map<String, Any?> {
        val result = rawArgs.toMutableMap()
        for (param in tool.parameters) {
            if (param.name in result && result[param.name] != null) continue

            // 1. Try endpointValue aliases
            if (param.name.equals("endpointValue", ignoreCase = true)) {
                val candidate = rawArgs["endpointValue"] ?: rawArgs["recipientId"]
                    ?: rawArgs["contactId"] ?: rawArgs["endpoint"] ?: rawArgs["id"]
                if (candidate != null) {
                    result[param.name] = candidate
                    continue
                }
            }

            // 2. Try case-insensitive or substring matching
            val matchedEntry = rawArgs.entries.firstOrNull { (k, v) ->
                v != null && (
                    param.name.endsWith(k, ignoreCase = true) ||
                    k.endsWith(param.name, ignoreCase = true) ||
                    param.name.equals(k, ignoreCase = true)
                )
            }
            if (matchedEntry != null) {
                result[param.name] = matchedEntry.value
            } else if (param.isRequired) {
                // Default value heuristics for well-known parameters
                if (param.name.equals("contactType", ignoreCase = true)) {
                    result[param.name] = "INDIVIDUAL"
                }
            }
        }
        return result
    }

    private fun parseToolCallFromText(
        text: String,
        tools: List<AppFunctionMetadata>,
    ): LlmResponsePart.ToolCall? {
        val jsonMatch = Regex("""\{[\s\S]*"name"[\s\S]*\}""").find(text)
        if (jsonMatch != null) {
            try {
                val element = Json.parseToJsonElement(jsonMatch.value).jsonObject
                val functionName = element["name"]?.jsonPrimitive?.contentOrNull
                if (functionName != null) {
                    val matchedTool = matchTool(functionName, tools)
                    if (matchedTool != null) {
                        val argsObj = (element["parameters"] ?: element["arguments"])?.jsonObject
                        val args = argsObj?.mapValues { (_, v) ->
                            v.jsonPrimitive.contentOrNull ?: v.toString()
                        } ?: emptyMap()
                        return LlmResponsePart.ToolCall(
                            packageName = matchedTool.packageName,
                            functionId = matchedTool.id,
                            arguments = args,
                            callId = UUID.randomUUID().toString(),
                        )
                    }
                }
            } catch (_: Exception) {}
        }
        return null
    }

    private fun filterRelevantTools(
        tools: List<AppFunctionMetadata>,
        query: String,
        maxTools: Int = 10,
    ): List<AppFunctionMetadata> {
        if (tools.size <= maxTools) return tools

        val queryWords = query.lowercase().split(Regex("""\W+""")).filter { it.length > 2 }
        fun scoreTool(tool: AppFunctionMetadata): Int {
            var score = 0
            val toolText = "${tool.id} ${tool.description} ${tool.packageName}".lowercase()
            for (word in queryWords) {
                if (toolText.contains(word)) score += 10
            }
            if (toolText.contains("call") || toolText.contains("contact") || toolText.contains("message") || toolText.contains("chat")) {
                score += 2
            }
            return score
        }

        return tools.sortedByDescending { scoreTool(it) }.take(maxTools)
    }

    private fun getSystemInstruction(): String {
        val currentDate = LocalDate.now().toString()
        return """
            You are an AI assistant running on Android. Today's date is $currentDate.
            Always reply to the user using concise, friendly, natural human language. Never respond to the user with raw JSON or structured code blocks unless explicitly asked to write code.
            When a user asks you to perform actions using an app, call the appropriate function tool.
            When a user asks you to generate an image, call the generateImage tool. After generateImage completes, confirm to the user in a natural sentence (for example: "I have generated that image for you.").
            When a user asks you to generate an image and use it in an app (for example, setting a chat wallpaper or attaching an image to a note), first call generateImage, then call the target app function passing the returned imageUri.
        """.trimIndent()
    }

    private fun processTurnFallback(
        history: List<GemmaTurn>,
        tools: List<AppFunctionMetadata>,
        modelName: String,
    ): List<LlmResponsePart> {
        val lastTurn = history.lastOrNull() ?: return listOf(LlmResponsePart.Text("No input provided."))

        if (lastTurn is GemmaTurn.ToolResult) {
            val searchContactOutput = lastTurn.outputs.firstOrNull {
                it.functionId.contains("searchContacts", ignoreCase = true)
            }

            if (searchContactOutput != null) {
                val parsed = parseContactSearchResult(searchContactOutput.result)
                if (parsed != null) {
                    val (endpointValue, displayName) = parsed
                    val originalUserText = history.filterIsInstance<GemmaTurn.User>().lastOrNull()?.text ?: ""

                    if (extractCallTarget(originalUserText) != null) {
                        val makeCallTool = tools.firstOrNull { it.id.contains("makeCall", ignoreCase = true) }
                        if (makeCallTool != null) {
                            val callId = UUID.randomUUID().toString()
                            return listOf(
                                LlmResponsePart.Text("Calling $displayName..."),
                                LlmResponsePart.ToolCall(
                                    packageName = makeCallTool.packageName,
                                    functionId = makeCallTool.id,
                                    arguments = mapOf("endpointValue" to endpointValue),
                                    callId = callId,
                                ),
                            )
                        }
                    }

                    val msgInfo = extractMessageTargetAndBody(originalUserText)
                    if (msgInfo != null) {
                        val sendMessageTool = tools.firstOrNull { it.id.contains("sendMessage", ignoreCase = true) }
                        if (sendMessageTool != null) {
                            val callId = UUID.randomUUID().toString()
                            val body = msgInfo.second.ifBlank { "Hello" }
                            return listOf(
                                LlmResponsePart.ToolCall(
                                    packageName = sendMessageTool.packageName,
                                    functionId = sendMessageTool.id,
                                    arguments = mapOf(
                                        "endpointValue" to endpointValue,
                                        "messageBody" to body,
                                    ),
                                    callId = callId,
                                ),
                            )
                        }
                    }

                    return listOf(
                        LlmResponsePart.Text("Found contact: $displayName (ID: $endpointValue)"),
                    )
                } else {
                    return listOf(
                        LlmResponsePart.Text("No contacts found matching your query."),
                    )
                }
            }

            val makeCallOutput = lastTurn.outputs.firstOrNull {
                it.functionId.contains("makeCall", ignoreCase = true)
            }
            val hasUserCallIntent = history.filterIsInstance<GemmaTurn.User>().any { extractCallTarget(it.text) != null }
            if (makeCallOutput != null && hasUserCallIntent) {
                val originalUserText = history.filterIsInstance<GemmaTurn.User>().lastOrNull()?.text ?: ""
                val target = extractCallTarget(originalUserText)
                val targetName = target?.replaceFirstChar { it.uppercase() } ?: "the contact"
                return listOf(
                    LlmResponsePart.Text("I have placed a call to $targetName."),
                )
            }

            val sendMessageOutput = lastTurn.outputs.firstOrNull {
                it.functionId.contains("sendMessage", ignoreCase = true)
            }
            val hasUserMsgIntent = history.filterIsInstance<GemmaTurn.User>().any { extractMessageTargetAndBody(it.text) != null }
            if (sendMessageOutput != null && hasUserMsgIntent) {
                val originalUserText = history.filterIsInstance<GemmaTurn.User>().lastOrNull()?.text ?: ""
                val msgInfo = extractMessageTargetAndBody(originalUserText)
                val targetName = msgInfo?.first?.replaceFirstChar { it.uppercase() } ?: "the contact"
                return listOf(
                    LlmResponsePart.Text("Message has been sent to $targetName."),
                )
            }

            val summaries = lastTurn.outputs.map { output ->
                "${output.functionId}: ${output.result}"
            }
            return listOf(
                LlmResponsePart.Text("Processed tool outputs: ${summaries.joinToString(", ")}"),
            )
        }

        if (lastTurn is GemmaTurn.User) {
            val userText = lastTurn.text.trim()

            val matchedTool = findMatchingTool(userText, tools)
            if (matchedTool != null) {
                val callId = UUID.randomUUID().toString()
                val arguments = extractToolArguments(userText, matchedTool)
                return listOf(
                    LlmResponsePart.ToolCall(
                        packageName = matchedTool.packageName,
                        functionId = matchedTool.id,
                        arguments = arguments,
                        callId = callId,
                    ),
                )
            }

            return listOf(
                LlmResponsePart.Text("Response from Gemma 4 ($modelName) for: $userText"),
            )
        }

        return listOf(LlmResponsePart.Text(""))
    }

    private fun parseContactSearchResult(result: String): Pair<String, String>? {
        return try {
            val element = Json.parseToJsonElement(result)
            if (element is JsonArray && element.isNotEmpty()) {
                val obj = element[0].jsonObject
                val endpointValue = obj["endpointValue"]?.jsonPrimitive?.contentOrNull
                val displayName = obj["contactDisplayName"]?.jsonPrimitive?.contentOrNull
                    ?: obj["endpointDisplayName"]?.jsonPrimitive?.contentOrNull
                    ?: endpointValue
                if (endpointValue != null) {
                    Pair(endpointValue, displayName ?: endpointValue)
                } else null
            } else if (element is JsonObject) {
                val endpointValue = element["endpointValue"]?.jsonPrimitive?.contentOrNull
                val displayName = element["contactDisplayName"]?.jsonPrimitive?.contentOrNull
                    ?: element["endpointDisplayName"]?.jsonPrimitive?.contentOrNull
                    ?: endpointValue
                if (endpointValue != null) {
                    Pair(endpointValue, displayName ?: endpointValue)
                } else null
            } else null
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse contact search result", e)
            null
        }
    }

    private fun findMatchingTool(
        userText: String,
        tools: List<AppFunctionMetadata>,
    ): AppFunctionMetadata? {
        val lower = userText.lowercase()

        if (extractCallTarget(userText) != null) {
            val searchTool = tools.firstOrNull { it.id.contains("searchContacts", ignoreCase = true) }
            if (searchTool != null) return searchTool
            val callTool = tools.firstOrNull { it.id.contains("makeCall", ignoreCase = true) }
            if (callTool != null) return callTool
        }

        if (extractMessageTargetAndBody(userText) != null) {
            val searchTool = tools.firstOrNull { it.id.contains("searchContacts", ignoreCase = true) }
            if (searchTool != null) return searchTool
            val sendTool = tools.firstOrNull { it.id.contains("sendMessage", ignoreCase = true) }
            if (sendTool != null) return sendTool
        }

        if (extractContactSearchQuery(userText) != null) {
            val searchTool = tools.firstOrNull { it.id.contains("searchContacts", ignoreCase = true) }
            if (searchTool != null) return searchTool
        }

        return tools.firstOrNull { tool ->
            val simpleName = tool.id.substringAfterLast("#").substringAfterLast(".").lowercase()
            val functionName = tool.id.substringAfterLast(".").lowercase()
            val toolName = toolConverter.getToolName(tool).lowercase()

            lower.contains(simpleName) ||
                lower.contains(functionName) ||
                lower.contains(toolName) ||
                matchesCamelCaseWords(simpleName, lower)
        }
    }

    private fun extractToolArguments(
        userText: String,
        tool: AppFunctionMetadata,
    ): Map<String, Any?> {
        val args = mutableMapOf<String, Any?>()
        val callTarget = extractCallTarget(userText)
        val msgInfo = extractMessageTargetAndBody(userText)
        val searchQuery = extractContactSearchQuery(userText)

        for (param in tool.parameters) {
            val paramLower = param.name.lowercase()
            when {
                paramLower == "query" -> {
                    args[param.name] = callTarget ?: msgInfo?.first ?: searchQuery ?: userText
                }
                paramLower == "contacttype" -> {
                    args[param.name] = "INDIVIDUAL"
                }
                paramLower == "endpointvalue" -> {
                    args[param.name] = callTarget ?: msgInfo?.first ?: userText
                }
                paramLower == "messagebody" || paramLower == "message" -> {
                    args[param.name] = msgInfo?.second ?: userText
                }
                param.name.contains("prompt", ignoreCase = true) -> {
                    val cleanedPrompt = userText.replace(
                        Regex("""^(?:generate|create|draw)\s+(?:an?\s+)?(?:image|picture)\s+(?:of\s+)?""", RegexOption.IGNORE_CASE),
                        "",
                    ).trim()
                    args[param.name] = cleanedPrompt.ifBlank { userText }
                }
                param.name.contains("query", ignoreCase = true) || param.name.contains("text", ignoreCase = true) -> {
                    args[param.name] = userText
                }
            }
        }
        return args
    }

    private fun matchesCamelCaseWords(simpleName: String, lowerText: String): Boolean {
        val words = simpleName.split(Regex("(?=[A-Z])")).map { it.lowercase() }.filter { it.length > 2 }
        return words.isNotEmpty() && words.all { lowerText.contains(it) }
    }

    companion object {
        private const val TAG = "GemmaProvider"

        private val CALL_PATTERNS = listOf(
            Regex("""\b(?:make|place)\s+a\s+call\s+to\s+([a-zA-Z0-9\s]+)""", RegexOption.IGNORE_CASE),
            Regex("""\b(?:call|phone|dial)\s+([a-zA-Z0-9\s]+)""", RegexOption.IGNORE_CASE),
        )

        private val MESSAGE_PATTERNS = listOf(
            Regex("""\b(?:send\s+(?:a\s+)?message\s+to|text|message)\s+([a-zA-Z0-9]+)\s+(?:saying|with|:)\s*(.*)""", RegexOption.IGNORE_CASE),
            Regex("""\b(?:send\s+(?:a\s+)?message\s+to|text|message)\s+([a-zA-Z0-9]+)\s+(.*)""", RegexOption.IGNORE_CASE),
            Regex("""\b(?:send\s+(?:a\s+)?message\s+to|text|message)\s+([a-zA-Z0-9]+)""", RegexOption.IGNORE_CASE),
        )

        private val SEARCH_PATTERNS = listOf(
            Regex("""\b(?:search\s+(?:contacts?\s+(?:for\s+)?)?|find\s+contact\s+|lookup\s+)([a-zA-Z0-9]+)""", RegexOption.IGNORE_CASE),
        )

        fun extractCallTarget(text: String): String? {
            for (pattern in CALL_PATTERNS) {
                val match = pattern.find(text)
                if (match != null) {
                    val rawTarget = match.groupValues[1].trim()
                    val cleaned = rawTarget.replace(Regex("""\b(?:please|now|on|at)\b.*$""", RegexOption.IGNORE_CASE), "").trim()
                    if (cleaned.isNotBlank()) return cleaned
                }
            }
            return null
        }

        fun extractMessageTargetAndBody(text: String): Pair<String, String>? {
            for (pattern in MESSAGE_PATTERNS) {
                val match = pattern.find(text)
                if (match != null) {
                    val target = match.groupValues[1].trim()
                    val body = if (match.groupValues.size > 2) match.groupValues[2].trim() else ""
                    if (target.isNotBlank()) return Pair(target, body)
                }
            }
            return null
        }

        fun extractContactSearchQuery(text: String): String? {
            for (pattern in SEARCH_PATTERNS) {
                val match = pattern.find(text)
                if (match != null) {
                    val target = match.groupValues[1].trim()
                    if (target.isNotBlank()) return target
                }
            }
            return null
        }
    }
}

/** Represents a conversation turn in Gemma 4 local history. */
sealed class GemmaTurn {
    data class User(val text: String) : GemmaTurn()
    data class ToolResult(val outputs: List<ToolOutput>) : GemmaTurn()
    data class Assistant(val parts: List<LlmResponsePart>) : GemmaTurn()
}
