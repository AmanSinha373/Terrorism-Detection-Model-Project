pipeline {
    agent any

    environment {
        // Define any environment variables, for example Docker image name
        IMAGE_NAME = "terrorism_detection_api"
    }

    stages {
        stage('Checkout') {
            steps {
                // Replace with your repository URL and branch if needed
                git url: 'https://github.com/AmanSinha373/Terrorism-Detection-Model-Project.git', branch: 'master'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${IMAGE_NAME}:${env.BUILD_ID}")
                }
            }
        }

        stage('Test') {
            steps {
                script {
                    // Optionally, run tests inside the Docker container.
                    // This example just runs the container and prints the output.
                    dockerImage.inside {
                        sh 'pytest'  // If you have tests; otherwise, you can run any validation command.
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    // Push the image to your Docker registry
                    // Ensure you have configured Docker credentials in Jenkins (replace 'docker-credentials-id' with your ID)
                    docker.withRegistry('', 'amansinha373') {
                        dockerImage.push()
                    }
                }
            }
        }
    }

    post {
        always {
            echo "Pipeline execution completed."
        }
    }
}
