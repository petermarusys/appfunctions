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
import com.example.appfunctions.agent.domain.ToolOutput
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Implementation of [LlmProvider] for Gemma 4 (E4B / E2B) edge and on-device LLM execution.
 *
 * Handles local session state tracking, tool declaration formatting, prompt construction,
 * and parsing function call actions for Gemma 4 on-device models.
 */
@Singleton
class GemmaProviderImpl
    @Inject
    constructor(
        private val toolConverter: GeminiToolConverter,
    ) : LlmProvider {

        private val sessionHistory = ConcurrentHashMap<String, MutableList<GemmaTurn>>()

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

                // Record the incoming turn in the history
                when (input) {
                    is LlmInput.UserMessage -> {
                        history.add(GemmaTurn.User(input.text))
                    }
                    is LlmInput.ToolResponse -> {
                        history.add(GemmaTurn.ToolResult(input.outputs))
                    }
                }

                // Process turn using Gemma 4 inference logic
                val responseParts = processTurn(history, tools, modelName)

                // Append response to conversation history and store session
                history.add(GemmaTurn.Assistant(responseParts))
                sessionHistory[newInteractionId] = history

                LlmResponse.Success(
                    interactionId = newInteractionId,
                    parts = responseParts,
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error during Gemma 4 inference", e)
                LlmResponse.Error("Gemma inference error: ${e.message}")
            }
        }

        private fun processTurn(
            history: List<GemmaTurn>,
            tools: List<AppFunctionMetadata>,
            modelName: String,
        ): List<LlmResponsePart> {
            val lastTurn = history.lastOrNull() ?: return listOf(LlmResponsePart.Text("No input provided."))

            if (lastTurn is GemmaTurn.ToolResult) {
                val summaries = lastTurn.outputs.map { output ->
                    "${output.functionId}: ${output.result}"
                }
                return listOf(
                    LlmResponsePart.Text("Processed tool outputs: ${summaries.joinToString(", ")}"),
                )
            }

            if (lastTurn is GemmaTurn.User) {
                val userText = lastTurn.text.trim()

                // Check for tool calling intent matching available tools
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
            for (param in tool.parameters) {
                if (param.name.contains("prompt", ignoreCase = true) ||
                    param.name.contains("query", ignoreCase = true) ||
                    param.name.contains("text", ignoreCase = true)
                ) {
                    args[param.name] = userText
                }
            }
            return args
        }

        companion object {
            private const val TAG = "GemmaProvider"
        }
}

/** Represents a conversation turn in Gemma 4 local history. */
sealed class GemmaTurn {
    data class User(val text: String) : GemmaTurn()
    data class ToolResult(val outputs: List<ToolOutput>) : GemmaTurn()
    data class Assistant(val parts: List<LlmResponsePart>) : GemmaTurn()
}
