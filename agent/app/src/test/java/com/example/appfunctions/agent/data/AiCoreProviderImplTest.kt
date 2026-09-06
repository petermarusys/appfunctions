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
import androidx.appfunctions.metadata.AppFunctionComponentsMetadata
import androidx.appfunctions.metadata.AppFunctionMetadata
import androidx.appfunctions.metadata.AppFunctionParameterMetadata
import androidx.appfunctions.metadata.AppFunctionResponseMetadata
import androidx.appfunctions.metadata.AppFunctionStringTypeMetadata
import com.example.appfunctions.agent.domain.LlmInput
import com.example.appfunctions.agent.domain.LlmResponse
import com.example.appfunctions.agent.domain.LlmResponsePart
import com.example.appfunctions.agent.domain.ToolOutput
import io.mockk.every
import io.mockk.mockkStatic
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class AiCoreProviderImplTest {
    private val toolConverter = GeminiToolConverter()
    private lateinit var provider: AiCoreProviderImpl

    @Before
    fun setUp() {
        mockkStatic(Log::class)
        every { Log.d(any(), any()) } returns 0
        every { Log.w(any<String>(), any<String>()) } returns 0
        every { Log.e(any(), any(), any()) } returns 0
        every { Log.e(any(), any()) } returns 0
        provider = AiCoreProviderImpl(toolConverter)
    }

    @Test
    fun generateResponse_userMessage_returnsSuccess() = runBlocking {
        val response =
            provider.generateResponse(
                previousInteractionId = null,
                input = LlmInput.UserMessage("Hello AICore"),
                tools = emptyList(),
                apiKey = "",
                modelName = "gemini-nano",
            )

        assertTrue(response is LlmResponse.Success)
        val success = response as LlmResponse.Success
        assertNotNull(success.interactionId)
        val textParts = success.parts.filterIsInstance<LlmResponsePart.Text>()
        assertTrue(textParts.isNotEmpty())
    }

    @Test
    fun generateResponse_multiTurn_maintainsSession() = runBlocking {
        val firstResponse =
            provider.generateResponse(
                previousInteractionId = null,
                input = LlmInput.UserMessage("First turn"),
                tools = emptyList(),
                apiKey = "",
                modelName = "gemini-nano",
            ) as LlmResponse.Success

        val secondResponse =
            provider.generateResponse(
                previousInteractionId = firstResponse.interactionId,
                input = LlmInput.UserMessage("Second turn"),
                tools = emptyList(),
                apiKey = "",
                modelName = "gemini-nano",
            )

        assertTrue(secondResponse is LlmResponse.Success)
    }

    @Test
    fun generateResponse_toolExecution_handlesToolCalls() = runBlocking {
        val tool =
            AppFunctionMetadata(
                id = "com.example.app.sendNote",
                packageName = "com.example.app",
                isEnabled = true,
                schema = null,
                description = "Sends a note",
                parameters =
                    listOf(
                        AppFunctionParameterMetadata(
                            name = "text",
                            isRequired = true,
                            dataType = AppFunctionStringTypeMetadata(isNullable = false),
                            description = "Note text",
                        ),
                    ),
                response =
                    AppFunctionResponseMetadata(
                        valueType = AppFunctionStringTypeMetadata(isNullable = false),
                        description = "",
                    ),
                components = AppFunctionComponentsMetadata(emptyMap()),
            )

        val response =
            provider.generateResponse(
                previousInteractionId = null,
                input = LlmInput.UserMessage("Please sendNote with message"),
                tools = listOf(tool),
                apiKey = "",
                modelName = "gemini-nano",
            ) as LlmResponse.Success

        val toolCalls = response.parts.filterIsInstance<LlmResponsePart.ToolCall>()
        assertEquals(1, toolCalls.size)
        assertEquals("com.example.app.sendNote", toolCalls[0].functionId)
        assertEquals("com.example.app", toolCalls[0].packageName)
    }

    @Test
    fun generateResponse_toolResponse_returnsProcessedText() = runBlocking {
        val response =
            provider.generateResponse(
                previousInteractionId = null,
                input =
                    LlmInput.ToolResponse(
                        listOf(
                            ToolOutput(
                                functionId = "com.example.app.sendNote",
                                callId = "call123",
                                result = "{\"status\":\"ok\"}",
                            ),
                        ),
                    ),
                tools = emptyList(),
                apiKey = "",
                modelName = "gemini-nano",
            ) as LlmResponse.Success

        val textParts = response.parts.filterIsInstance<LlmResponsePart.Text>()
        assertTrue(textParts.any { it.text.contains("sendNote") })
    }
}
