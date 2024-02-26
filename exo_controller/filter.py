import numpy as np
import cupy as cp
import sys


class MichaelFilter:
    def __init__(
        self,
        output_smoothing=True,
        additinal_filter=True,
        only_additional_filter=False,
        weight_prediction_impact_on_regression=0.7,  # the lower the more smooth but less reactive ( 0.75 is good)
        weight_additional_filter=1,  # the higher the smoother but delayed / less reactive (3 is good)
        degree_regressions=1,
        buffer_length=5,  # (5 is good)
        skip_predictions=1,
        num_fingers=2,
    ):
        self.last_val = []
        self.last_avg = []
        self.output_smoothing = output_smoothing
        self.additinal_filter = additinal_filter
        self.only_additional_filter = only_additional_filter
        # a new prediciton is genereated, which is between the regresion prediction and the ai prediction. This
        # weight regulats the influence of the ai prediction on the new prediciton.
        self.weight_prediction_impact_on_regression = (
            weight_prediction_impact_on_regression
        )
        # the distance ( how much changes from last to this point) gets divided by this and then normalized
        # how much of the change willl be taken -> the hiher the weight the less change will be taken
        self.weight_additional_filter = weight_additional_filter
        # which degree of polynomial regression shuld be used.
        # degree 1 has to be used
        self.degree_regressions = degree_regressions
        # how many last predictions/averages should be regarded f.e for regression
        self.buffer_length = buffer_length
        # how many samples in the future should be predicted
        self.skip_predictions = skip_predictions
        # num fingers is how many fingers we want to predict i.e how many values one prediction has
        self.num_fingers = num_fingers

    def parabola(self, x, a):
        return a * x**2

    def modified_relu(self, x):
        # Segment 1: y = 0 for x <= 0.1
        # Segment 2: y increases with a slope of 1 from x = 0.1 until y = 1
        # Segment 3: y = 1 for the rest
        #TODO hier sollte ich 0.005 nochmal erhöhen zu 0.01 oder mittelteil zusätzlich machen bei dem nicht steigung 1 sondern wo ende dannn auch parabel aber anders herum also zwischen 0.5 und 1 dann auch limitiert
        a1 = 0.005 / (
            0.05**2
        )  # value for parabola       y_value / (x_value ** 2)      -> when i want the parabola to be 0.02 at 0.02 then y and x value shoudl be 0.02
        y = np.where(x <= 0.05, self.parabola(x, a1), np.where(x < 1, x, 1))

        #comment following line out if want to be faster than hand, use the line if you want smoother predictions
        y = np.where(y>0.3, -0.4286 * y**2 + 0.1286 * y + 0.3, y)

        return y

    def normalize(self, matrix):
        return matrix / (np.linalg.norm(matrix) + 1e-8)


    def filter(self, prediction):
        self.last_avg.append(prediction)
        self.last_val.append(prediction)

        if (len(self.last_avg) == 0) or len(self.last_avg) == 1:
            return prediction

        if self.only_additional_filter:
            new = self.last_avg[-2] + (
                self.modified_relu(np.abs(prediction - self.last_avg[-2]))
                / (self.weight_additional_filter)
            ) * (prediction - self.last_avg[-2])
            self.last_avg[-1] = new
            prediction = new
            if len(self.last_avg) > self.buffer_length:
                self.last_avg.pop(0)

            return prediction

        if 1 < len(self.last_avg) < self.buffer_length:
            new = self.last_avg[-2] + (
                self.modified_relu(np.abs(prediction - self.last_avg[-2]))
                / (self.weight_additional_filter)
            ) * (prediction - self.last_avg[-2])
            self.last_avg[-1] = new
            prediction = new
            return prediction

        for i in range(self.num_fingers):
            last_joint_preds = []
            last_joint_avgs = []
            for l in range(self.buffer_length - 1):
                last_joint_preds.append(self.last_val[l][i])
                last_joint_avgs.append(self.last_avg[l][i])

            mymodel3 = np.poly1d(
                np.polyfit(
                    np.linspace(1, self.buffer_length - 1, self.buffer_length - 1),
                    last_joint_preds,
                    self.degree_regressions,
                )
            )
            res = []
            for u in range(self.buffer_length):
                res.append(mymodel3(u))
            # plt.scatter(np.linspace(1,buffer_length,buffer_length), p, color='red')
            # plt.plot(np.linspace(1,buffer_length,buffer_length), res, color='green')
            pred_reg = (
                mymodel3(self.buffer_length + self.skip_predictions)
                + prediction[i] * self.weight_prediction_impact_on_regression
            ) / (self.weight_prediction_impact_on_regression + 1)

            mymodel = np.poly1d(
                np.polyfit(
                    np.linspace(1, self.buffer_length - 1, self.buffer_length - 1),
                    last_joint_avgs,
                    self.degree_regressions,
                )
            )
            avg_reg = (
                mymodel(self.buffer_length + self.skip_predictions)
                + prediction[i] * self.weight_prediction_impact_on_regression
            ) / (self.weight_prediction_impact_on_regression + 1)
            res2 = []
            for u in range(self.buffer_length):
                res2.append(mymodel(u))
            # plt.plot(np.linspace(1, buffer_length, buffer_length), res2, color='blue')
            # plt.scatter(np.linspace(1, buffer_length, buffer_length), q, color='black')
            # plt.show()

            crozzcoeff = np.abs(np.corrcoef(res2, res)[0, 1])
            if np.isnan(crozzcoeff):
                crozzcoeff = 1

            ges_pred = (avg_reg * crozzcoeff + pred_reg) / (crozzcoeff + 1)
            # ges_pred = ((mymodel3(buffer_length) + mymodel(buffer_length) + prediction[i][j]*var2) / (2 + var2))

            self.last_avg[-1][i] = ges_pred

        if self.additinal_filter == True:
            new = self.last_avg[-2] + (
                self.modified_relu(np.abs(self.last_avg[-1] - self.last_avg[-2]))
                / self.weight_additional_filter
            ) * (self.last_avg[-1] - self.last_avg[-2])
            #TODO hier überprüfen ob zu grosser sprung von einem zum anderen sample ist (mehr als 0.5 ?? ) wenn ja dann sollte ich es einfach lassen ?
            self.last_avg[-1] = new
            self.last_val.pop(0)
            self.last_avg.pop(0)
            prediction = new
            return prediction

        self.last_val.pop(0)
        self.last_avg.pop(0)
        prediction = self.last_avg[-1]
        return prediction

