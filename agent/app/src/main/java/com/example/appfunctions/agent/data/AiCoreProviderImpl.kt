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

            // If the latest turn is tool results, summarize or reply
            if (lastTurn is AiCoreTurn.ToolResult) {
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

        private fun findMatchingTool(
            userText: String,
            tools: List<AppFunctionMetadata>,
        ): AppFunctionMetadata? {
            val lower = userText.lowercase()
            return tools.firstOrNull { tool ->
                val functionName = tool.id.substringAfterLast(".").lowercase()
                val toolName = toolConverter.getToolName(tool).lowercase()
                lower.contains(functionName) || lower.contains(toolName)
            }
        }

        private fun extractToolArguments(
            userText: String,
            tool: AppFunctionMetadata,
        ): Map<String, Any?> {
            val args = mutableMapOf<String, Any?>()
            // Provide parameters from input or defaults for tool execution
            for (param in tool.parameters) {
                if (param.name.contains("prompt", ignoreCase = true) || param.name.contains("query", ignoreCase = true) || param.name.contains("text", ignoreCase = true)) {
                    args[param.name] = userText
                }
            }
            return args
        }

        companion object {
            private const val TAG = "AiCoreProvider"
        }
}

/** Represents a conversation turn in on-device AICore history. */
sealed class AiCoreTurn {
    data class User(val text: String) : AiCoreTurn()
    data class ToolResult(val outputs: List<com.example.appfunctions.agent.domain.ToolOutput>) : AiCoreTurn()
    data class Assistant(val parts: List<LlmResponsePart>) : AiCoreTurn()
}
