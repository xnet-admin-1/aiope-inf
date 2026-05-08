#!/usr/bin/env bash
set -e

# Attempt to find Java
JAVA_EXEC=$(command -v java 2>/dev/null || echo "")
if [ -z "$JAVA_EXEC" ]; then
    echo "Java not found. Please install Java 11 or higher."
    exit 1
fi

# Run Gradle wrapper if it exists
if [ -f "gradle/wrapper/gradle-wrapper.jar" ]; then
    exec "$JAVA_EXEC" -cp gradle/wrapper/gradle-wrapper.jar org.gradle.wrapper.GradleWrapperMain "$@"
else
    echo "Gradle wrapper jar not found. Downloading..."
    curl -fsSL https://services.gradle.org/distributions/gradle-8.5-all.zip -o /tmp/gradle-wrapper.zip
    mkdir -p gradle/wrapper
    # We can't extract without unzip, so we'll create a simple script
    cat > gradle/wrapper/gradle-wrapper.jar << 'INNER_EOF'
echo "Please download gradle manually or use Android Studio"
INNER_EOF
    chmod +x gradle/wrapper/gradle-wrapper.jar
    echo "Unable to download gradle wrapper without unzip command"
    echo "Please use 'gradle build' directly or install unzip"
    exit 1
fi
