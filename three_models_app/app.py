import json
from pathlib import Path

import matplotlib.pyplot as plt
style_path = Path(__file__).parent / "custom.mplstyle"
plt.style.use(style_path)
import numpy as np
from shiny import App, render, ui, reactive

# data = pd.read_csv(Path(__file__).parent / "weekly_seasonality.csv")

# #turn 0->1 into 0-7 (for days of the week)
# data['x'] = 7*data['x']/data['x'].max()
# data['y'] = 5*data['y']/data['y'].max()

# extra_weeks = pd.DataFrame(
#     {
#         "y": np.concatenate((data['y'].values, data['y'].values)),
#         "x": np.concatenate((data['x'].values+7, data['x'].values+ (2*7)))
#         }
# )

# data = pd.concat([data, extra_weeks])
# data = data.sort_values("x")

# Replace "your_file_path.csv" with the actual path to your CSV file
file_path = Path(__file__).parent / "weekly_seasonality.csv"

# Reading CSV file into a numpy array
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # Assuming the first row is a header

# Assuming 'x' is in the first column and 'y' is in the second column after the header
x = data[:, 0]
y = data[:, 1]

# Transform 'x' and 'y'
x_transformed = 7 * x / x.max()
y_transformed = 5 * y / y.max()

# Creating extra weeks
x_extra_weeks = np.concatenate((x_transformed + 7, x_transformed + (2 * 7)))
y_extra_weeks = np.concatenate((y_transformed, y_transformed))

# Combine original and extra weeks
x_combined = np.concatenate((x_transformed, x_extra_weeks))
y_combined = np.concatenate((y_transformed, y_extra_weeks))

# Creating a combined array for sorting purposes
combined = np.column_stack((x_combined, y_combined))

# Sorting by 'x' values
sorted_indices = combined[:, 0].argsort()
data = combined[sorted_indices]

print(data)

def dummy(x, start, width = 1):
    x_mod = (x % 7)
    # Create a boolean array where True is set for elements within the specified range
    condition = (x_mod >= start) & (x_mod < start + width)

    # Convert the boolean array to an integer array (True becomes 1, False becomes 0)
    return condition.astype(int)

def dummy_model(
        x,
        C_1,
        C_2,
        C_3,
        C_4,
        C_5,
        C_6,
        C_7,
):
    return (
        C_1 * dummy(x, start= 0) + 
        C_2 * dummy(x, start= 1) + 
        C_3 * dummy(x, start= 2) + 
        C_4 * dummy(x, start= 3) + 
        C_5 * dummy(x, start= 4) + 
        C_6 * dummy(x, start= 5) + 
        C_7 * dummy(x, start= 6)
    )


def rbf(x, width, center):
    x_mod = x % 7
    center_mod = center % 7
    
    # Original Gaussian
    gauss = np.exp(-((x_mod - center_mod)**2) / (2 * width))
    
    # Gaussian shifted by +7
    gauss_plus = np.exp(-((x_mod - (center_mod + 7))**2) / (2 * width))
    
    # Gaussian shifted by -7
    gauss_minus = np.exp(-((x_mod - (center_mod - 7))**2) / (2 * width))
    
    # Sum the contributions
    return gauss + gauss_plus + gauss_minus

def rbf_model(
        x,
        C_1,
        C_2,
        C_3,
        C_4,
        C_5,
        C_6,
        C_7,
        width,
):
    return (
        C_1 * rbf(x, width=width, center = 0) + 
        C_2 * rbf(x, width=width, center = 1) + 
        C_3 * rbf(x, width=width, center = 2) + 
        C_4 * rbf(x, width=width, center = 3) + 
        C_5 * rbf(x, width=width, center = 4) + 
        C_6 * rbf(x, width=width, center = 5) + 
        C_7 * rbf(x, width=width, center = 6)
    )


def fourier_component(periods, func, order):
    return getattr(np, func)(2 * np.pi * periods * order)

def fourier_model(
        x,
        C_cos_1,
        C_sin_1,
        C_cos_2,
        C_sin_2
    ):
    return (
        C_cos_1 * fourier_component(x/7, "cos", 1) + 
        C_sin_1 * fourier_component(x/7, "sin", 1) + 
        C_cos_2 * fourier_component(x/7, "cos", 2) + 
        C_sin_2 * fourier_component(x/7, "sin", 2)
    )


app_ui = ui.page_fluid(
            ui.navset_tab(
                ui.nav_panel(
                    "Dummy",
                    ui.page_fillable(
                        ui.card(
                            ui.output_plot("plot_dummy"),
                        ),
                        ui.card(
                            ui.layout_columns(
                                ui.card(
                                    ui.input_switch("best_fit_dummy", "Show best fit?", True),  
                                    ui.input_slider("C_1_dummy", "Monday", 0, 5, value = 1.9 , step = 0.1),
                                    ui.input_slider("C_2_dummy", "Tuesday", 0, 5, value = 3.8, step = 0.1),
                                    ui.input_slider("C_3_dummy", "Wednesday", 0, 5, value = 4.1, step = 0.1),
                                ),
                                ui.card(
                                    ui.input_slider("C_4_dummy", "Thursday", 0, 5, value = 3.0, step = 0.1),
                                    ui.input_slider("C_5_dummy", "Friday", 0, 5, value = 3.3, step = 0.1),
                                    ui.input_slider("C_6_dummy", "Saturday", 0, 5, value = 2.8, step = 0.1),
                                    ui.input_slider("C_7_dummy", "Sunday", 0, 5, value = 1.8, step = 0.1),
                                )
                            ),
                        ),
                        
                    ),
                ),
                ui.nav_panel(
                    "Radial Basis Function",
                    ui.page_fillable(
                        ui.card(
                            ui.output_plot("plot_rbf"),
                        ),
                        ui.card(
                            ui.layout_columns(
                                ui.card(
                                    ui.input_switch("best_fit_rbf", "Show best fit?", True),  
                                    ui.input_slider("width", "Width", 0.01, 0.5, value = 0.3 , step = 0.01),
                                    ui.input_slider("C_1_rbf", "Monday", 0, 5, value =  1.2, step = 0.1),
                                    ui.input_slider("C_2_rbf", "Tuesday", 0, 5, value = 1.9, step = 0.1),
                                    ui.input_slider("C_3_rbf", "Wednesday", 0, 5, value = 3.7, step = 0.1),
                                ),
                                ui.card(
                                    ui.input_slider("C_4_rbf", "Thursday", 0, 5, value = 2.4, step = 0.1),
                                    ui.input_slider("C_5_rbf", "Friday", 0, 5, value = 2.2, step = 0.1),
                                    ui.input_slider("C_6_rbf", "Saturday", 0, 5, value = 2.6, step = 0.1),
                                    ui.input_slider("C_7_rbf", "Sunday", 0, 5, value = 1.8, step = 0.1),
                                )
                            ),
                        ),
                        
                    ),
                ),
                ui.nav_panel(
                    "Fourier components",
                    ui.page_fillable(
                        ui.card(
                            ui.output_plot("plot_fourier"),
                        ),
                        ui.layout_columns(
                            ui.card(
                                ui.input_switch("best_fit_fourier", "Show best fit?", True),  
                                ui.input_slider("n_order", "N order", 1, 10, value = 2 , step = 1),
                            ),
                            ui.card(
                                ui.input_slider("C_cos_1", "Cosine 1st order", -1, 1, value = -0.78 , step = 0.1),
                                ui.input_slider("C_sin_1", "Sine 1st order", -1, 1, value = 0.38, step = 0.1),
                                ui.input_slider("C_cos_2", "Cosine 2nd order", -1, 1, value = -0.5, step = 0.1),
                                ui.input_slider("C_sin_2", "Sine 2nd order", -1, 1, value = -0.1, step = 0.1),
                            ),
                        ),
                    ),
                        
                ),
                selected = 'Radial Basis Function'
            ),
)


# app_ui = ui.page_fluid(
#             ui.navset_tab(
#                 ui.nav_panel(
#                     "Dummy",
#                     ui.page_sidebar(
#                         ui.sidebar(
#                             ui.input_switch("best_fit_dummy", "Show best fit?", False),  
#                             ui.input_slider("C_1_dummy", "Monday", 0, 5, value = 2.0 , step = 0.1),
#                             ui.input_slider("C_2_dummy", "Tuesday", 0, 5, value = 3.7, step = 0.1),
#                             ui.input_slider("C_3_dummy", "Wednesday", 0, 5, value = 4.2, step = 0.1),
#                             ui.input_slider("C_4_dummy", "Thursday", 0, 5, value = 3.0, step = 0.1),
#                             ui.input_slider("C_5_dummy", "Friday", 0, 5, value = 3.0, step = 0.1),
#                             ui.input_slider("C_6_dummy", "Saturday", 0, 5, value = 2.8, step = 0.1),
#                             ui.input_slider("C_7_dummy", "Sunday", 0, 5, value = 1.0, step = 0.1),
#                             position="right",
#                             bg="#f8f8f8",
#                             open="open"
#                         ),
#                         ui.output_plot("plot_dummy"),
#                     ),
#                 ),
#                 ui.nav_panel(
#                     "Radial Basis Function",
#                     ui.page_sidebar(
#                         ui.sidebar(
#                             ui.input_switch("best_fit_rbf", "Show best fit?", False),  
#                             ui.input_slider("width", "Width", 0.01, 0.5, value = 0.3 , step = 0.01),
#                             ui.input_slider("C_1_rbf", "Monday", 0, 5, value = 2.2 , step = 0.1),
#                             ui.input_slider("C_2_rbf", "Tuesday", 0, 5, value = 3.7, step = 0.1),
#                             ui.input_slider("C_3_rbf", "Wednesday", 0, 5, value = 2.4, step = 0.1),
#                             ui.input_slider("C_4_rbf", "Thursday", 0, 5, value = 1.7, step = 0.1),
#                             ui.input_slider("C_5_rbf", "Friday", 0, 5, value = 2.3, step = 0.1),
#                             ui.input_slider("C_6_rbf", "Saturday", 0, 5, value = 1.4, step = 0.1),
#                             ui.input_slider("C_7_rbf", "Sunday", 0, 5, value = 1.0, step = 0.1),
#                             position="right",
#                             bg="#f8f8f8",
#                             open="open"
#                         ),
#                         ui.output_plot("plot_rbf"),
#                     ),
#                 ),
#                 ui.nav_panel(
#                     "Fourier components",
#                     "2 Content"
#                 ),
#                 selected = 'Radial Basis Function'
#             )
# )


# ui.page_sidebar(
#     ui.sidebar(
#         ui.input_switch("best_fit", "Show best fit?", False),  
#         ui.input_slider("C_0", "Monday", 0, 5, value = 2.0 , step = 0.1),
#         ui.input_slider("C_1", "Tuesday", 0, 5, value = 3.7, step = 0.1),
#         ui.input_slider("C_2", "Wednesday", 0, 5, value = 4.2, step = 0.1),
#         ui.input_slider("C_3", "Thursday", 0, 5, value = 3.0, step = 0.1),
#         ui.input_slider("C_4", "Friday", 0, 5, value = 3.0, step = 0.1),
#         ui.input_slider("C_5", "Saturday", 0, 5, value = 2.8, step = 0.1),
#         ui.input_slider("C_6", "Sunday", 0, 5, value = 1.0, step = 0.1),
#         position="right",
#         bg="#f8f8f8",
#         open="open"
#     ),
#     ui.output_plot("plot_dummy"),
# ),

def server(input, output, session):
    @output
    @render.plot(alt="A scatterplot")
    def plot():
        
        fig, ax = plt.subplots()
        plt.title("Test")
        x = data[:,0]
        y = data[:,1]
        ax.scatter(x, y)

        return fig

    @output
    @render.plot(alt="A scatterplot")
    def plot_dummy():
        C_1 = input.C_1_dummy()
        C_2 = input.C_2_dummy()
        C_3 = input.C_3_dummy()
        C_4 = input.C_4_dummy()
        C_5 = input.C_5_dummy()
        C_6 = input.C_6_dummy()
        C_7 = input.C_7_dummy()

        fig, axs = plt.subplots(1,1)

        x = data[:,0]
        y = data[:,1]
        axs.scatter(x, y, color = 'slategrey', s= 5)

        top_x = 7*3
        xs = np.linspace(0,top_x, 1000)

        alpha = 0.6
        axs.fill_between(xs, C_1 * dummy(xs, start= 0), alpha = alpha)
        axs.fill_between(xs, C_2 * dummy(xs, start= 1), alpha = alpha)
        axs.fill_between(xs, C_3 * dummy(xs, start= 2), alpha = alpha)
        axs.fill_between(xs, C_4 * dummy(xs, start= 3), alpha = alpha)
        axs.fill_between(xs, C_5 * dummy(xs, start= 4), alpha = alpha)
        axs.fill_between(xs, C_6 * dummy(xs, start= 5), alpha = alpha)
        axs.fill_between(xs, C_7 * dummy(xs, start= 6), alpha = alpha)

        total = dummy_model(
            xs,
            C_1, 
            C_2, 
            C_3, 
            C_4, 
            C_5, 
            C_6, 
            C_7
        )


        


        pred_y = dummy_model(
            x,
            C_1,
            C_2,
            C_3, 
            C_4, 
            C_5, 
            C_6, 
            C_7
        )

        mse = np.mean((y - pred_y)**2)

        axs.plot(
            xs, 
            total,
            color = 'fuchsia',
            ls = '--',
            label = f'Your model MSE: {mse:.3f}'
        )

        if input.best_fit_dummy():
            # plot best fit (cannot be bothered to refit here, just use values from notebook)

            C_1 = 2.36
            C_2 = 3.62
            C_3 = 3.93
            C_4 = 3.25
            C_5 = 3.29
            C_6 = 3.04
            C_7 = 2.10

            total = dummy_model(
                xs,
                C_1, 
                C_2, 
                C_3, 
                C_4, 
                C_5, 
                C_6, 
                C_7
            )

            pred_y = dummy_model(
                x,
                C_1,
                C_2,
                C_3, 
                C_4, 
                C_5, 
                C_6, 
                C_7
            )

            mse = np.mean((y - pred_y)**2)

            axs.plot(
                xs, 
                total,
                color = 'darkorchid',
                label = f'Best fit MSE: {mse:.3f}'
            )

        
        axs.set_xlim(0,top_x)
        axs.set_ylim(0,6.5)

        # Define the days of the week
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Set xticks and labels
        plt.xticks(np.arange(0, max(xs) + 1, 1), [days_of_week[int(day) % 7] for day in np.arange(0, max(xs) + 1, 1)], rotation=90)

        axs.set_ylabel('y')

        axs.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=1)
        
        return fig
    
    @output
    @render.plot(alt="A scatterplot")
    def plot_rbf():
        width = input.width()
        C_1 = input.C_1_rbf()
        C_2 = input.C_2_rbf()
        C_3 = input.C_3_rbf()
        C_4 = input.C_4_rbf()
        C_5 = input.C_5_rbf()
        C_6 = input.C_6_rbf()
        C_7 = input.C_7_rbf()

        fig, axs = plt.subplots(1,1)

        x = data[:,0]
        y = data[:,1]
        axs.scatter(x, y, color = 'slategrey', s= 5)

        top_x = 7*3
        xs = np.linspace(0,top_x, 1000)

        alpha = 0.6
        axs.fill_between(xs, C_1 * rbf(xs, width = width, center= 0), alpha = alpha)
        axs.fill_between(xs, C_2 * rbf(xs, width = width, center= 1), alpha = alpha)
        axs.fill_between(xs, C_3 * rbf(xs, width = width, center= 2), alpha = alpha)
        axs.fill_between(xs, C_4 * rbf(xs, width = width, center= 3), alpha = alpha)
        axs.fill_between(xs, C_5 * rbf(xs, width = width, center= 4), alpha = alpha)
        axs.fill_between(xs, C_6 * rbf(xs, width = width, center= 5), alpha = alpha)
        axs.fill_between(xs, C_7 * rbf(xs, width = width, center= 6), alpha = alpha)

        total = rbf_model(
            xs,
            C_1, 
            C_2, 
            C_3, 
            C_4, 
            C_5, 
            C_6, 
            C_7,
            width=width
        )


        

        

        pred_y = rbf_model(
            x,
            C_1,
            C_2,
            C_3, 
            C_4, 
            C_5, 
            C_6, 
            C_7,
            width=width
        )

        mse = np.mean((y - pred_y)**2)
        # sum_of_squares = np.sum((data['y'] - data['y'].mean())**2)

        # R2 = 1 - sum_of_residuals/sum_of_squares

        # axs.set_title(f"MSE: {mse:.3f}", size = 14)
        axs.plot(
                xs, 
                total,
                color = 'fuchsia',
                ls = '--',
                label = f'Your model MSE: {mse:.3f}'
            )

        if input.best_fit_rbf():
            # plot best fit (cannot be bothered to refit here, just use values from notebook)

            coeffs = fit_rbf()
            width = input.width()
            X_train = np.column_stack([rbf(xs, width = width, center =  center,) for center in range(1, 8)])
            y_to_plot = np.dot(X_train, coeffs)
            
            X = np.column_stack([rbf(x, width = width, center =  center,) for center in range(1, 8)])
            y_pred = np.dot(X, coeffs)

            mse = np.mean((y - y_pred)**2)

            axs.plot(
                xs, 
                y_to_plot,
                color = 'darkorchid',
                label = f'Best fit MSE: {mse:.3f}'
            )

        axs.set_xlim(0,top_x)
        # axs.set_ylim(0,6.5)

        # Define the days of the week
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Set xticks and labels
        plt.xticks(np.arange(0, max(xs) + 1, 1), [days_of_week[int(day) % 7] for day in np.arange(0, max(xs) + 1, 1)], rotation=90)

        axs.set_ylabel('y')

        axs.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=1)

        return fig
    
    @output
    @render.plot(alt="A scatterplot")
    def plot_fourier():
        C_cos_1 = input.C_cos_1()
        C_cos_2 = input.C_cos_2()
        C_sin_1 = input.C_sin_1()
        C_sin_2 = input.C_sin_2()


        fig, axs = plt.subplots(1,1)

        x = data[:,0]
        y = data[:,1]
        axs.scatter(x, y, color = 'slategrey', s= 5)

        top_x = 7*3
        xs = np.linspace(0,top_x, 1000)

        alpha = 0.6
        axs.plot(xs, C_cos_1 * fourier_component(xs/7, "cos", 1), alpha = alpha)
        axs.plot(xs, C_sin_1 * fourier_component(xs/7, "sin", 1), alpha = alpha)
        axs.plot(xs, C_cos_2 * fourier_component(xs/7, "cos", 2), alpha = alpha)
        axs.plot(xs, C_sin_2 * fourier_component(xs/7, "sin", 2), alpha = alpha)
        
        total = fourier_model(
            xs,
            C_cos_1,
            C_sin_1,
            C_cos_2,
            C_sin_2,
        ) + np.mean(y)


        

        

        pred_y = fourier_model(
            x,
            C_cos_1,
            C_sin_1,
            C_cos_2,
            C_sin_2,
        ) + np.mean(y)

        mse = np.mean((y - pred_y)**2)
        # sum_of_squares = np.sum((data['y'] - data['y'].mean())**2)

        # R2 = 1 - sum_of_residuals/sum_of_squares

        # axs.set_title(f"MSE: {mse:.3f}", size = 14)
        axs.plot(
                xs, 
                total,
                color = 'fuchsia',
                ls = '--',
                label = f'Your model MSE: {mse:.3f}'
        )

        if input.best_fit_fourier():
            # plot best fit (cannot be bothered to refit here, just use values from notebook)

            coeffs = fit_fourier()

            n_order = input.n_order()
            X_train = np.column_stack([
                fourier_component(xs/7, func, order)
                for order in range(1, n_order + 1)
                for func in ("sin", "cos")
            ])
            y_to_plot = np.dot(X_train, coeffs) + np.mean(y)
            
            X = get_fourier_X()
            y_pred = np.dot(X, coeffs) + np.mean(y)

            mse = np.mean((y - y_pred)**2)

            axs.plot(
                xs, 
                y_to_plot,
                color = 'darkorchid',
                label = f'Best fit MSE: {mse:.3f}'
            )

        axs.set_xlim(0,top_x)
        # axs.set_ylim(0,6.5)

        # Define the days of the week
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Set xticks and labels
        plt.xticks(np.arange(0, max(xs) + 1, 1), [days_of_week[int(day) % 7] for day in np.arange(0, max(xs) + 1, 1)], rotation=90)

        axs.set_ylabel('y')

        axs.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=1)
        print(fig)
        return fig
    
    @reactive.Calc
    def get_rbf_X():
        
        width = input.width()

        x = data[:,0]
        # Creating RBF features
        X = np.column_stack([rbf(x, width = width, center =  center) for center in range(1, 8)])
        
        return X
    
    @reactive.Calc
    def fit_rbf():
        X = get_rbf_X()
        
        to_fit_y = data[:,1].copy()

        # Perform OLS to solve for coefficients
        XtX = np.dot(X.T, X)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X.T, to_fit_y)
        beta = np.dot(XtX_inv, Xty)
        return beta
    
    @reactive.Calc
    def get_fourier_X():
        
        n_order = input.n_order()

        x = data[:,0]
        # Creating RBF features
        X = np.column_stack([
            fourier_component(x/7, func, order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        ])
        return X
    
    @reactive.Calc
    def fit_fourier():

        X = get_fourier_X()
        
        to_fit_y = data[:,1].copy()
        to_fit_y -= np.mean(to_fit_y)

        # Perform OLS to solve for coefficients
        XtX = np.dot(X.T, X)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X.T, to_fit_y)
        beta = np.dot(XtX_inv, Xty)

        return beta
    


        


app = App(app_ui, server, debug=False)
