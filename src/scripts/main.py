script = int(input("Which script do you want to run? "
                   "\n - IMDB Text Classifier [1]"
                   "\n - MNIST Image Classifier [2]"
                   "\n - YT Title predictor [3]"  # ToDo
                   "\n - Car Evaluation [4]"
                   "\n - Census Income [5]"
                   "\n - Stock Price Predictor [6]"
                   "\n - Student Performance [7]"
                   "\n - Cancer Predictor [8]"
                   "\n - K-Means Clustering [9]"  # ToDo
                   "\n - Tic Tac Toe [10]"
                   "\n: "
                   ))

match script:
    case 1:
        exec(open("neural_networks/imdb_text_classification.py").read())
    case 2:
        exec(open("neural_networks/mnist_image_classification.py").read())
    case 3:
        exec(open("neural_networks/yt_title_predictor.py").read())
    case 4:
        exec(open("supervised/knn/car-evaluation.py").read())
    case 5:
        exec(open("supervised/linear_regression/census_income.py").read())
    case 6:
        exec(open("supervised/linear_regression/predictor.py").read())
    case 7:
        exec(open("supervised/linear_regression/student-performance.py").read())
    case 8:
        exec(open("supervised/svm/cancer.py").read())
    case 9:
        exec(open("unsupervised/k-means-clustering.py").read())
    case 10:
        exec(open("tic-tac-toe.py").read())
