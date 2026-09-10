pipeline {
    agent any
    options { timeout(time: 20, unit: 'MINUTES') }
    environment { IMAGE = 'debray2523/auctionsense' }
    stages {
        stage('Checkout') { steps { checkout scm } }
        stage('Test') {
            steps {
                bat 'python -m venv .venv'
                bat '.venv\\Scripts\\python -m pip install --quiet --upgrade pip'
                bat '.venv\\Scripts\\python -m pip install --quiet -r requirements.txt'
                bat '.venv\\Scripts\\python -m pytest tests --junitxml=results.xml'
            }
            post { always { junit allowEmptyResults: true, testResults: 'results.xml' } }
        }
        stage('Build image') {
            when { anyOf { branch 'release/*'; branch 'hotfix/*'; branch 'main' } }
            steps {
                bat 'docker build -t %IMAGE%:%GIT_COMMIT:~0,7% .'
                bat 'docker run --rm %IMAGE%:%GIT_COMMIT:~0,7%'
            }
        }
        stage('Push image') {
            when { branch 'main' }
            steps {
                input message: 'Push this image to Docker Hub?', ok: 'Push'
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'U', passwordVariable: 'P')]) {
                    bat 'docker login -u %U% -p %P%'
                    bat 'for /f %%v in (VERSION) do docker tag %IMAGE%:%GIT_COMMIT:~0,7% %IMAGE%:%%v && docker push %IMAGE%:%%v'
                    bat 'docker logout'
                }
            }
        }
    }
}