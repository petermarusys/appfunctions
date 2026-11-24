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
package com.example.appfunctions.agent.ui.screens.main

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import com.example.appfunctions.agent.MainActivity.Companion.EXTRA_INSTRUMENTATION_ERROR
import com.example.appfunctions.agent.domain.troubleshoot.CheckAppFunctionPermissionUseCase
import com.example.appfunctions.agent.domain.troubleshoot.CheckMainlineVersionUseCase
import com.example.appfunctions.agent.domain.troubleshoot.CheckOsVersionUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

/**
 * ViewModel for the main screen shell.
 *
 * Manages the state for the app status dialog.
 */
@HiltViewModel
class MainViewModel
    @Inject
    constructor(
        private val checkAppFunctionPermissionUseCase: CheckAppFunctionPermissionUseCase,
        private val checkOsVersionUseCase: CheckOsVersionUseCase,
        savedStateHandle: SavedStateHandle,
    ) : ViewModel() {
        private val _uiState = MutableStateFlow(MainUiState())
        val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

        init {
            val error = savedStateHandle.get<String>(EXTRA_INSTRUMENTATION_ERROR)
            if (error != null) {
                _uiState.value =
                    _uiState.value.copy(
                        appStatus = AppStatus.InstrumentationFailed(error),
                        showDialog = true,
                    )
            } else {
                checkAppStatus()
            }
        }

        /** Checks the app status including support and permission. */
        fun checkAppStatus() {
            val isOsSupported = checkOsVersionUseCase()

            if (!isOsSupported) {
                _uiState.value =
                    _uiState.value.copy(
                        appStatus = AppStatus.NotSupported,
                        showDialog = true,
                    )
                return
            }

            val hasPermission = checkAppFunctionPermissionUseCase()
            _uiState.value =
                _uiState.value.copy(
                    appStatus =
                        if (hasPermission) AppStatus.PermissionGranted else AppStatus.PermissionMissing,
                    showDialog = true,
                )
        }

        /** Dismisses the status dialog. */
        fun dismissDialog() {
            _uiState.value = _uiState.value.copy(showDialog = false)
        }
    }

/** Sealed class representing the app status. */
sealed class AppStatus {
    object Idle : AppStatus()

    object PermissionGranted : AppStatus()

    object PermissionMissing : AppStatus()

    object NotSupported : AppStatus()

    data class InstrumentationFailed(val error: String) : AppStatus()
}

/** UI state for the main screen. */
data class MainUiState(
    val appStatus: AppStatus = AppStatus.Idle,
    val showDialog: Boolean = false,
)
