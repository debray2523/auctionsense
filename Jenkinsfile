pipeline {
    agent any
    options { timeout(time: 20, unit: 'MINUTES') }
    environment { IMAGE = 'debray2523/auctionsense' }
    stages {
        stage('Checkout') { steps { checkout scm } }
        stage('Build image') {
            steps { bat 'docker build -t %IMAGE%:%GIT_COMMIT:~0,7% .' }
        }
        stage('Test in container') {
            steps { bat 'docker run --rm %IMAGE%:%GIT_COMMIT:~0,7% python -m pytest tests' }
        }
        stage('Push image') {
            when { branch 'main' }
            steps {
                input message: 'Push this image to Docker Hub?', ok: 'Push'
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'U', passwordVariable: 'P')]) {
                    bat 'echo %P%| docker login -u %U% --password-stdin'
                    bat 'for /f %%v in (VERSION) do docker tag %IMAGE%:%GIT_COMMIT:~0,7% %IMAGE%:%%v && docker push %IMAGE%:%%v'
                    bat 'docker logout'
                }
            }
        }
        stage('Deploy to cluster') {
            when { branch 'main' }
            steps {
                bat 'for /f %%v in (VERSION) do kubectl set image deployment/auctionsense api=%IMAGE%:%%v'
                bat 'kubectl rollout status deployment/auctionsense --timeout=180s'
                bat 'kubectl get pods -l app=auctionsense'
            }
        }
    }
}