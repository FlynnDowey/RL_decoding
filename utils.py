import argparse
import matplotlib.pyplot as plt
from scipy.io import loadmat


def plot_BER(x_vars, y_sarsa, y_qlearn, bench, bfd, units, title, figure_name):
    plt.semilogy(x_vars, y_sarsa, marker='o', fillstyle='none', linestyle='--', linewidth=1.5, markersize=8, label='Tabular sarsa')
    plt.semilogy(x_vars, y_qlearn, marker='o', fillstyle='none', linestyle='--', linewidth=1.5, markersize=8, label='Tabular Q-learning')
    plt.semilogy(x_vars, bench, marker='o', fillstyle='none', linestyle='--', linewidth=1.5, markersize=8, label='Traditional decoder')
    # plt.semilogy(x_vars, y_param, marker='o', fillstyle='none', linestyle='--', linewidth=1.5, markersize=8, label='NN sarsa')
    plt.semilogy(x_vars, bfd, marker='o', fillstyle='none', linestyle='--', linewidth=1.5, markersize=8, label='BFD')
    plt.xlabel(units)
    plt.ylabel('BER')
    plt.title(title)
    plt.grid(visible=True, which='both', axis='y')
    plt.grid(visible=True, which='major', axis='x')
    plt.legend()
    plt.savefig('./figs/' + figure_name + '.png')
    plt.show()

_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    questions = sorted(_funcs.keys())
    parser.add_argument(
        "questions",
        choices=(questions + ["all"]),
        nargs="+",
        help="A question ID to run, or 'all'.",
    )
    args = parser.parse_args()
    for q in args.questions:
        if q == "all":
            for q in sorted(_funcs.keys()):
                start = f"== {q} "
                print("\n" + start + "=" * (80 - len(start)))
                run(q)

        else:
            run(q)