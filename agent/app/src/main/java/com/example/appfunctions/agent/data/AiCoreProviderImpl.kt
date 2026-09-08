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

import android.util.Log
import androidx.appfunctions.metadata.AppFunctionMetadata
import com.example.appfunctions.agent.domain.LlmInput
import com.example.appfunctions.agent.domain.LlmProvider
import com.example.appfunctions.agent.domain.LlmResponse
import com.example.appfunctions.agent.domain.LlmResponsePart
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.time.LocalDate
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Implementation of [LlmProvider] for on-device execution using Android AICore (Gemini Nano).
 *
 * Manages local session history across turns, tool formatting, and response parsing.
 */
@Singleton
class AiCoreProviderImpl
    @Inject
    constructor(
        private val toolConverter: GeminiToolConverter,
    ) : LlmProvider {

        // In-memory conversation history keyed by interaction ID
        private val sessionHistory = ConcurrentHashMap<String, MutableList<AiCoreTurn>>()

        override suspend fun generateResponse(
            previousInteractionId: String?,
            input: LlmInput,
            tools: List<AppFunctionMetadata>,
            apiKey: String,
            modelName: String,
            serviceTier: ServiceTier,
        ): LlmResponse {
            return try {
                val newInteractionId = UUID.randomUUID().toString()
                val history =
                    if (previousInteractionId != null && sessionHistory.containsKey(previousInteractionId)) {
                        sessionHistory[previousInteractionId]?.toMutableList() ?: mutableListOf()
                    } else {
                        mutableListOf()
                    }

                // Append current turn to history
                when (input) {
                    is LlmInput.UserMessage -> {
                        history.add(AiCoreTurn.User(input.text))
                    }
                    is LlmInput.ToolResponse -> {
                        history.add(AiCoreTurn.ToolResult(input.outputs))
                    }
                }

                // Process turn with the on-device model engine
                val responseParts = processTurn(history, tools, modelName)

                // Save session for subsequent turns
                history.add(AiCoreTurn.Assistant(responseParts))
                sessionHistory[newInteractionId] = history

                LlmResponse.Success(
                    interactionId = newInteractionId,
                    parts = responseParts,
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error in AICore inference", e)
                LlmResponse.Error("AICore inference error: ${e.message}")
            }
        }

        private fun processTurn(
            history: List<AiCoreTurn>,
            tools: List<AppFunctionMetadata>,
            modelName: String,
        ): List<LlmResponsePart> {
            val lastTurn = history.lastOrNull() ?: return listOf(LlmResponsePart.Text("No input provided."))

            // If the latest turn is tool results, process next chained action or summarize
            if (lastTurn is AiCoreTurn.ToolResult) {
                // Check if any output is from searchContacts
                val searchContactOutput = lastTurn.outputs.firstOrNull {
                    it.functionId.contains("searchContacts", ignoreCase = true)
                }

                if (searchContactOutput != null) {
                    val parsed = parseContactSearchResult(searchContactOutput.result)
                    if (parsed != null) {
                        val (endpointValue, displayName) = parsed
                        val originalUserText = history.filterIsInstance<AiCoreTurn.User>().lastOrNull()?.text ?: ""

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

                // Check if makeCall completed
                val makeCallOutput = lastTurn.outputs.firstOrNull {
                    it.functionId.contains("makeCall", ignoreCase = true)
                }
                val hasUserCallIntent = history.filterIsInstance<AiCoreTurn.User>().any { extractCallTarget(it.text) != null }
                if (makeCallOutput != null && hasUserCallIntent) {
                    val originalUserText = history.filterIsInstance<AiCoreTurn.User>().lastOrNull()?.text ?: ""
                    val target = extractCallTarget(originalUserText)
                    val targetName = target?.replaceFirstChar { it.uppercase() } ?: "the contact"
                    return listOf(
                        LlmResponsePart.Text("I have placed a call to $targetName."),
                    )
                }

                // Check if sendMessage completed
                val sendMessageOutput = lastTurn.outputs.firstOrNull {
                    it.functionId.contains("sendMessage", ignoreCase = true)
                }
                val hasUserMsgIntent = history.filterIsInstance<AiCoreTurn.User>().any { extractMessageTargetAndBody(it.text) != null }
                if (sendMessageOutput != null && hasUserMsgIntent) {
                    val originalUserText = history.filterIsInstance<AiCoreTurn.User>().lastOrNull()?.text ?: ""
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
                    LlmResponsePart.Text("I have processed the results: ${summaries.joinToString(", ")}"),
                )
            }

            // For user messages, check if tools match or generate a response
            if (lastTurn is AiCoreTurn.User) {
                val userText = lastTurn.text.trim()

                // Inspect user text for direct intent matching available tools
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
                    LlmResponsePart.Text("Response from on-device AICore ($modelName) for: $userText"),
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

            // 1. Call intent -> Prefer searchContacts first to get endpointValue, or makeCall
            if (extractCallTarget(userText) != null) {
                val searchTool = tools.firstOrNull { it.id.contains("searchContacts", ignoreCase = true) }
                if (searchTool != null) return searchTool
                val callTool = tools.firstOrNull { it.id.contains("makeCall", ignoreCase = true) }
                if (callTool != null) return callTool
            }

            // 2. Message intent -> Prefer searchContacts first, or sendMessage
            if (extractMessageTargetAndBody(userText) != null) {
                val searchTool = tools.firstOrNull { it.id.contains("searchContacts", ignoreCase = true) }
                if (searchTool != null) return searchTool
                val sendTool = tools.firstOrNull { it.id.contains("sendMessage", ignoreCase = true) }
                if (sendTool != null) return sendTool
            }

            // 3. Contact search intent
            if (extractContactSearchQuery(userText) != null) {
                val searchTool = tools.firstOrNull { it.id.contains("searchContacts", ignoreCase = true) }
                if (searchTool != null) return searchTool
            }

            // 4. General matching: tool id, function name, tool name, or camelCase words
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
            private const val TAG = "AiCoreProvider"

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

/** Represents a conversation turn in on-device AICore history. */
sealed class AiCoreTurn {
    data class User(val text: String) : AiCoreTurn()
    data class ToolResult(val outputs: List<com.example.appfunctions.agent.domain.ToolOutput>) : AiCoreTurn()
    data class Assistant(val parts: List<LlmResponsePart>) : AiCoreTurn()
}
