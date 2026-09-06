pipeline {
    agent any
    options { timeout(time: 15, unit: 'MINUTES') }
    stages {
        stage('Checkout') { steps { checkout scm } }
        stage('Info') {
            steps {
                bat 'git log -1 --oneline'
                bat 'dir'
            }
        }
        stage('Test') {
            steps {
                bat 'python --version'
                bat 'if exist requirements.txt pip install -r requirements.txt'
                bat 'if exist tests (python -m pytest tests) else (echo No tests folder yet)'
            }
        }
        stage('Deploy to UAT') {
            when { anyOf { branch 'release/*'; branch 'hotfix/*' } }
            steps { echo "Would deploy ${env.BRANCH_NAME} to UAT here" }
        }
        stage('Deploy to production') {
            when { branch 'main' }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                echo 'Would deploy to production here'
            }
        }
    }
}