from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROUNDING_FACTOR = 3
MU_VECTOR = [0, 0, 4, 0]
COV_ARRAY = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]

pio.templates.default = "simple_white"

Q_ONE_STRING = "Question (1)\n({mu}, {var})"
Q_FOUR_STRING = "\nQuestion (4)\nExpectation:\n {mu}\nCovariance:\n{cov}"
Q_SIX_STRING = "\nQuestion (6)\nThe values of (f1, f3) that gets the" \
               " maximum log-likelihood is:\n({f_1_ind}, {f_3_ind})"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univarient_estimator = UnivariateGaussian()
    samples_vector = np.random.normal(10, 1, 1000)
    univarient_estimator.fit(samples_vector)
    print(Q_ONE_STRING.format(mu=univarient_estimator.mu_, var=univarient_estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    x_axis = np.linspace(10, 1000, 100).astype(int)
    distance_from_mean = []
    for x in x_axis:
        sub_samples_vector = samples_vector[:x]
        distance_from_mean.append(abs(np.mean(sub_samples_vector) - univarient_estimator.mu_))

    go.Figure([go.Scatter(x=x_axis, y=distance_from_mean, mode='markers+lines', name=r'$\widehat\mu$'),
               go.Scatter(x=x_axis, y=[0] * len(x_axis), mode='lines', name=r'$\mu$')]).update_layout(
        title=r"$\text{(2) Estimation of absolute distance between the estimated -"
              r" and true value of the Expectation As Function Of Number Of Samples}$",
        xaxis_title="$m\\text{ - number of samples}$",
        yaxis_title="r${distanst from }\\mu$").show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_vector = univarient_estimator.pdf(samples_vector)
    fig = make_subplots(rows=1, cols=1, specs=[[{"rowspan": 1, "secondary_y": True}]]) \
        .add_trace(go.Scatter(x=samples_vector, y=pdf_vector, mode='markers', opacity=0.75)) \
        .add_trace(go.Histogram(x=samples_vector, y=pdf_vector, opacity=0.75, bingroup=1, nbinsx=100,
                                histnorm="probability density"))
    fig.update_layout(title_text="$\\text{(3) Histograms of PDF samples - It can be seen from the graph that the PDF}"
                                 "\\sim\\mathcal{N}\\left(10,1\\right)\\text{ as expected}\\\\$") \
        .update_yaxes(title_text="$\\text{PDF}\\\\$", secondary_y=False, row=1, col=1) \
        .update_yaxes(showgrid=False, row=1, col=1, showticklabels=True) \
        .update_xaxes(showgrid=False, title_text="Samples Values", row=1, col=1) \
        .update_layout(showlegend=False)
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array(MU_VECTOR)
    cov = np.array(COV_ARRAY)
    samples_vector_multi_d = np.random.multivariate_normal(mu, cov, 1000)
    multi_d_estimator = MultivariateGaussian()
    multi_d_estimator.fit(samples_vector_multi_d)
    print(Q_FOUR_STRING.format(mu=multi_d_estimator.mu_, cov=multi_d_estimator.cov_))

    # Question 5 - Likelihood evaluation
    x_axis = np.linspace(-10, 10, 200)
    log_likelihood_vector = []
    multi_d_estimator.pdf(samples_vector_multi_d)
    MultivariateGaussian.log_likelihood(multi_d_estimator.mu_,
                                        multi_d_estimator.cov_, samples_vector_multi_d)
    f_1, f_3 = [], []
    for i in range(x_axis.shape[0]):
        for j in range(x_axis.shape[0]):
            f_1.append(x_axis[i])
            f_3.append(x_axis[j])
            log_likelihood_vector.append(
                MultivariateGaussian.log_likelihood(np.array([x_axis[i], 0, x_axis[j], 0]), cov,
                                                    samples_vector_multi_d))
    go.Figure(go.Heatmap(x=f_3, y=f_1, z=log_likelihood_vector)).update_layout(
        title="$\\text{(5) Heatmap of the log_likelihood - It can be seen from the most "
              "likelihood is around f1 = 0 and f3 = 4 which make sense because "
              "the expectation is [0,0,4,0]}\\\\$", xaxis_title="r${f_3}$", yaxis_title="r${f_1}$").show()

    # Question 6 - Maximum likelihood
    max_like_index = np.argmax(log_likelihood_vector)
    print(Q_SIX_STRING.format(f_1_ind=round(f_1[max_like_index], ROUNDING_FACTOR),
                              f_3_ind=round(f_3[max_like_index], ROUNDING_FACTOR)))


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    # test_multivariate_gaussian()
