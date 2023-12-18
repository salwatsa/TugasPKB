using Serialization, Statistics, LinearAlgebra

# Calculate average
function calculate_rowwise_average(df)
    return DataFrame(mean(Matrix(df), dims=2), [:average])
end

# Compute Euclidean distance between features and means
function vectorized_euclidean_distance(X, mu)
    numclass = size(mu)[3]
    X = repeat(X, outer=[1, 1, numclass])
    subtracted_vector = sqrt.(sum((X .- mu).^2, dims=2))
    return subtracted_vector
end

#  classify by Euclidean distance
function classify_by_distance_euclidean(X, mu)
    num_instance = size(X)[1]
    mu_vec = repeat(mu, outer=[num_instance, 1, 1])
    dist_vec = vectorized_euclidean_distance(X, mu_vec)
    min_vector = argmin(dist_vec, dims=3)
    min_index = @. get_min_index(min_vector)
    return min_index
end

function get_min_index(X)
    return X[3]
end

function confusion_matrix(truth, preds)
    class = unique(truth)
    class_size = length(class)
    valuation = zeros(Int, class_size, class_size)
    for i = 1:class_size
        for j = 1:class_size
            valuation[i, j] = sum((truth .== class[i]) .& (preds .== class[j]))
        end
    end
    return valuation
end

# Cascade classify based on mean values
function cascade_classifier(dataset)
    class = unique(dataset[:, end])
    class_index = size(dataset, 2)
    col_size = size(dataset, 2)
    feature_size = col_size - 1
    mu_vec = zeros(Float16, 1, feature_size, length(class))

    for i = 1:length(class)
        c = class[i]
        current_class_pos = (dataset[:, class_index] .- c) .< Float16(0.1)
        current_df = dataset[current_class_pos, 1:class_index-1]
        current_df = Float32.(current_df)
        mu = mean(current_df, dims=1)
        mu_vec[1, :, i] = mu
    end
    return mu_vec
end

# Cascade classify based on Euclidean distance and per feature
function cascade_classify_euclidean(dataset, mu)
    class_index = size(dataset, 2)
    feature_size = size(mu, 2)

    # make prediction using Euclidean distance and per feature
    preds = zeros(Int, size(dataset, 1), feature_size)

    for i = 1:feature_size
        current_feature = dataset[:, i]
        current_feature = reshape(current_feature, (size(current_feature, 1), 1))
        current_mu = reshape(mu[1, i, :], (1, 1, size(mu, 3)))
        current_pred = classify_by_distance_euclidean(current_feature, current_mu)
        if i == 1
            preds = current_pred
        else
            preds = hcat(preds, current_pred)
        end
    end

    truth = dataset[:, class_index]
    temp = hcat(dataset, preds)
    return truth, preds
end

# Compute correctness based on the confusion matrix
function true_correctness(valuation)
    return sum(diag(valuation)) / sum(valuation)
end

# Display confusion matrix and correctness
function display_confusion_and_correctness(valuation, correctness)
    println("Confusion Matrix:")
    println(valuation)
    println("Correctness: ", correctness)
end

# Main execution
dataset = deserialize("data_9m.mat")
mu_vector = cascade_classifier(dataset)
truths, preds = cascade_classify_euclidean(dataset, mu_vector)
valuation = confusion_matrix(truths, preds)
correctness = true_correctness(valuation)

display_confusion_and_correctness(valuation, correctness)
