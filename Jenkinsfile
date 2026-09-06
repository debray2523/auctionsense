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
                bat 'python -m venv .venv'
                bat '.venv\\Scripts\\python -m pip install --quiet --upgrade pip'
                bat '.venv\\Scripts\\python -m pip install --quiet -r requirements.txt'
                bat '.venv\\Scripts\\python -m pytest tests --junitxml=results.xml'
            }
            post { always { junit allowEmptyResults: true, testResults: 'results.xml' } }
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