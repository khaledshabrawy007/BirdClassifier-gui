import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_COLORS = ['#2563EB', '#DC2626'] 
BOUNDARY_COLOR = '#16A34A'


def plot_decision_boundary(ax, weights, bias, X_train, y_train,
                           X_test, y_test, feature1, feature2,
                           class1, class2, use_bias: bool = True):
   
    w1, w2 = weights

  
    all_X = np.vstack([X_train, X_test])
    x_min, x_max = all_X[:, 0].min(), all_X[:, 0].max()
    margin = (x_max - x_min) * 0.1

    xi_vals = np.linspace(x_min - margin, x_max + margin, 300)

    if abs(w2) > 1e-10:
        if use_bias:
            xj_vals = -(w1 * xi_vals + bias) / w2
        else:
            xj_vals = -(w1 * xi_vals) / w2
        ax.plot(xi_vals, xj_vals, color=BOUNDARY_COLOR, linewidth=2,
                label='Decision Boundary', zorder=3)
    else:
        
        vx = -bias / w1 if abs(w1) > 1e-10 else 0
        ax.axvline(x=vx, color=BOUNDARY_COLOR, linewidth=2,
                   label='Decision Boundary', zorder=3)

  
    for cls_val, color in zip([1, -1], CLASS_COLORS):
        mask = y_train == cls_val
        lbl = class1 if cls_val == 1 else class2
        ax.scatter(X_train[mask, 0], X_train[mask, 1],
                   c=color, alpha=0.75, s=60, edgecolors='white',
                   linewidth=0.5, label=f'{lbl} (train)', zorder=2)


    for cls_val, color in zip([1, -1], CLASS_COLORS):
        mask = y_test == cls_val
        lbl = class1 if cls_val == 1 else class2
        ax.scatter(X_test[mask, 0], X_test[mask, 1],
                   c='none', edgecolors=color, s=80, linewidth=1.5,
                   marker='s', label=f'{lbl} (test)', zorder=2)

    ax.set_xlabel(feature1, fontsize=10)
    ax.set_ylabel(feature2, fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def plot_confusion_matrix(ax, cm, class1, class2):
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([class1, class2], fontsize=10)
    ax.set_yticklabels([class1, class2], fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')


def plot_training_curve(ax, history, algorithm: str):
    ax.plot(history, color='#7C3AED', linewidth=2)
    if algorithm == 'Adaline':
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_title('MSE per Epoch', fontsize=11)
    else:
        ax.set_ylabel('Misclassifications', fontsize=10)
        ax.set_title('Errors per Epoch', fontsize=11)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.grid(True, alpha=0.3)


def build_result_figure(weights, bias, X_train, y_train, X_test, y_test,
                        y_pred, cm, accuracy, feature1, feature2,
                        class1, class2, algorithm: str,
                        training_history: list, use_bias: bool = True):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(
        f'{algorithm} | {class1} vs {class2} | {feature1} × {feature2} | '
        f'Accuracy: {accuracy * 100:.1f}%',
        fontsize=12, fontweight='bold'
    )

    plot_decision_boundary(axes[0], weights, bias,
                           X_train, y_train, X_test, y_test,
                           feature1, feature2, class1, class2, use_bias)
    axes[0].set_title('Decision Boundary', fontsize=11)

    plot_confusion_matrix(axes[1], cm, class1, class2)
    axes[1].set_title('Confusion Matrix', fontsize=11)

    plt.tight_layout()
    return fig
