package com.aiope.inf

import android.app.*
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat

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
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        modelManager = ModelManager.getInstance(this)
        acquireWakeLock()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_SERVER -> {
                val port = intent.getIntExtra(EXTRA_PORT, 8008)
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

        val stopIntent = PendingIntent.getService(
            this, 0,
            Intent(this, InferenceService::class.java).apply { action = ACTION_STOP_SERVER },
            PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("AIOPE Inference")
            .setContentText("API server running on :$port")
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setOngoing(true)
            .setSilent(true)
            .setPriority(NotificationCompat.PRIORITY_MIN)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopIntent)
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }

    private fun stopServer() {
        server?.stop()
        server = null
        modelManager?.unload()
        releaseWakeLock()
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "aiope:inference").apply {
            acquire()
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "AIOPE Inference",
                NotificationManager.IMPORTANCE_MIN
            ).apply {
                description = "Local LLM inference engine"
                setShowBadge(false)
                setSound(null, null)
                enableVibration(false)
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopServer()
        super.onDestroy()
    }
}
