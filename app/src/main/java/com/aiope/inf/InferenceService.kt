package com.aiope.inf

import android.app.*
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

/**
 * Foreground service for running inference and OpenAI API server
 * in the background. Keeps the process alive during long inference tasks.
 */
class InferenceService : Service() {

    companion object {
        const val CHANNEL_ID = "aiope_inf_channel"
        const val NOTIFICATION_ID = 1001
        const val ACTION_START_SERVER = "com.aiope.inf.START_SERVER"
        const val ACTION_STOP_SERVER = "com.aiope.inf.STOP_SERVER"
        const val EXTRA_PORT = "port"
    }

    private var modelManager: ModelManager? = null
    private var server: OpenAIServer? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        modelManager = ModelManager(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_SERVER -> {
                val port = intent.getIntExtra(EXTRA_PORT, 8080)
                startServer(port)
            }
            ACTION_STOP_SERVER -> {
                stopServer()
                stopSelf()
            }
        }
        return START_STICKY
    }

    private fun startServer(port: Int) {
        server = OpenAIServer(modelManager!!).also { it.start(port) }

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("AIOPE Inference")
            .setContentText("OpenAI API server running on :$port")
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setOngoing(true)
            .addAction(
                android.R.drawable.ic_media_pause,
                "Stop",
                PendingIntent.getService(
                    this, 0,
                    Intent(this, InferenceService::class.java).apply {
                        action = ACTION_STOP_SERVER
                    },
                    PendingIntent.FLAG_IMMUTABLE
                )
            )
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }

    private fun stopServer() {
        server?.stop()
        server = null
        modelManager?.unload()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "AIOPE Inference Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Local LLM inference engine"
            }
            getSystemService(NotificationManager::class.java)
                .createNotificationChannel(channel)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopServer()
        super.onDestroy()
    }
}
