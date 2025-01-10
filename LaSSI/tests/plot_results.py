import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme, scale_y_log10, scale_color_brewer, \
    element_rect, ylim, coord_cartesian, scale_x_continuous, geom_point


def main():
    pd.set_option('display.max_columns', None)

    data = pd.read_csv('./jan10-benchmark.csv')
    data = data.sort_values(by='Dataset')
    data = data.replace(0, np.nan) # For generating/loading meuDB
    averaged_data = data.groupby('Dataset', as_index=False).mean(numeric_only=True)

    melted_data = averaged_data.melt(id_vars=['Dataset'], var_name='Phase', value_name='Time')
    melted_data['Phase'] = pd.Categorical(melted_data['Phase'], categories=list(data.columns[1:]), ordered=True)
    print(list(data.columns[1:]))

    plot = (
        ggplot(melted_data, aes(x='Dataset', y='Time', color='Phase')) +
        geom_line(size=1) +
        geom_point(aes(shape="Phase"), size=2) +
        scale_x_continuous(breaks=sorted(data['Dataset'].unique())) +
        scale_y_log10(breaks=[10 ** x for x in range(-5, 4)],
                      labels=lambda l: ["{:.0e}".format(v).replace("+0", "+").replace("-0", "+") for v in l]) +
        scale_color_brewer(type='qual', palette='Dark2') +
        labs(title='LaSSI Phase Execution Times vs. Number of Sentences',
             x='Number of sentences',
             y='Time (seconds, log scale)',
             color='Phase') +
        theme_minimal() +
        theme(plot_background=element_rect(fill='white'))
        # coord_cartesian(ylim=(1e-5, 1e+4))
    )
    plot.save('performance_metrics_plot.png', width=800, height=600, limitsize=False)

if __name__ == "__main__":
    main()