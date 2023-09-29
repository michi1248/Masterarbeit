from typing import List, Optional, Tuple

import numba

import numpy as np
import torch
from numba.typed import List as NumbaList
from torchmetrics import PearsonCorrCoef


@numba.njit
def _normalize_numpy(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix)


@torch.jit.script
def _normalize_torch(matrix) -> torch.Tensor:
    return matrix / torch.linalg.norm(matrix)


def _append_to_tensor(tensor: torch.Tensor, to_append: torch.Tensor):
    return torch.concat([tensor, to_append[None, ...]], dim=0)


def _pop_first_in_tensor(tensor: Optional[torch.Tensor]):
    return tensor[1:]


@numba.njit
def _additional_filter(last_avg: List, prediction, weight_additional_filter: float):
    new = last_avg[-2] + (
        _normalize_numpy(np.sqrt(np.square(prediction - last_avg[-2]))) / weight_additional_filter
    ) * (prediction - last_avg[-2])
    last_avg[-1] = new

    return new


@numba.njit
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0], deg + 1))
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_


@numba.njit
def _fit_x(a, b):
    # linalg solves ax = b
    return np.linalg.lstsq(a, b)[0]


@numba.jit
def fit_poly(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


@numba.njit
def eval_polynomial(P, x):
    """
    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0].  Uses Horner's Method.
    """
    result = 0
    for coeff in P:
        result = x * result + coeff
    return result


# def _normalize_cupy(matrix) -> cupy.ndarray:
#     return matrix / cupy.linalg.norm(matrix)
#
#
# def _append_to_cupy(array: cupy.ndarray, to_append: cupy.ndarray):
#     return cupy.concatenate([array, to_append[None, ...]], axis=0)
#
#
# def _pop_first_in_cupy(array: Optional[cupy.ndarray]):
#     return array[1:]


class MichaelFilter:
    """Michael's filter implementation

    Parameters
    ----------
    window_size : int
        The window size for the moving average
    threshold : float
        The threshold for the additional filter
    additional_filter : bool
        Whether to use the additional filter or not
    only_additional_filter : bool
        Whether to only use the additional filter or not
    weight_prediction_impact_on_regression : float
        The weight of the prediction impact on the regression (0.5 means that the prediction has the same impact as the
        regression)
    weight_additional_filter : float
        The weight of the additional filter (higher means that the additional filter has a higher impact)
    buffer_length : int
        The buffer length for the additional filter
    skip_predictions : int
        The number of predictions to skip before using the additional filter
    cuda : bool
        Whether to use cuda or not
    num_outputs : int
        The number of outputs


    Returns
    -------
    np.ndarray
        The filtered data

    Notes
    -----
        Original ("naive") implementation made by Michael März (michael.maerz@fau.de)
        during his bachelor thesis at the Nsquared-Lab at
        Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) in 2022.

        Subsequent improvements made by Raul C. Sîmpetru (raul.simpetru@fau.de).

    """

    def __init__(
        self,
        threshold: float = 0.5,
        additional_filter: bool = True,
        only_additional_filter: bool = False,
        weight_prediction_impact_on_regression: float = 0.5,
        weight_additional_filter: float = 3,
        buffer_length: int = 15,
        skip_predictions: int = 1,
        cuda: bool = False,
        num_outputs: int = 60,
    ):
        self.threshold = threshold
        self.additional_filter = additional_filter
        self.only_additional_filter = only_additional_filter
        self.weight_prediction_impact_on_regression = weight_prediction_impact_on_regression
        self.weight_additional_filter = weight_additional_filter
        self.buffer_length = buffer_length
        self.skip_predictions = skip_predictions
        self.cuda = cuda
        self.num_outputs = num_outputs

        self._last_avg = None
        self._last_val = None

        self._xs = torch.arange(0, self.buffer_length, 1, device="cpu" if not self.cuda else "cuda")
        self._pearson = PearsonCorrCoef(device="cpu" if not self.cuda else "cuda", num_outputs=self.num_outputs)
        self._A = torch.ones(
            (self.num_outputs, self.buffer_length, 2),
            device="cpu" if not self.cuda else "cuda",
            dtype=torch.float32,
        )
        self._A[:, :, 1] = self._xs

    def naive_filter(self, prediction: torch.FloatTensor) -> np.ndarray:
        prediction = prediction.T

        if self._last_avg is None and self._last_val is None:
            self._last_avg = [prediction]
            self._last_val = [prediction]
        else:
            self._last_avg.append(prediction)
            self._last_val.append(prediction)

        if len(self._last_avg) <= 1:
            return prediction

        if self.only_additional_filter:
            new = self._last_avg[-2] + (
                _normalize_numpy(np.sqrt(np.square(prediction - self._last_avg[-2]))) / self.weight_additional_filter
            ) * (prediction - self._last_avg[-2])
            self._last_avg[-1] = new
            if len(self._last_avg) > self.buffer_length:
                self._last_avg.pop(0)

            return new

        if 1 < len(self._last_avg) < self.buffer_length:
            new = self._last_avg[-2] + (
                _normalize_numpy(np.sqrt(np.square(prediction - self._last_avg[-2]))) / self.weight_additional_filter
            ) * (prediction - self._last_avg[-2])
            self._last_avg[-1] = new

            return new

        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                last_joint_preds = []
                last_joint_avgs = []
                for k in range(self.buffer_length - 1):
                    last_joint_preds.append(self._last_val[k][i][j])
                    last_joint_avgs.append(self._last_avg[k][i][j])

                mymodel3 = np.poly1d(
                    np.polyfit(
                        np.linspace(1, self.buffer_length - 1, self.buffer_length - 1),
                        last_joint_preds,
                        1,
                    )
                )
                res = []
                for u in range(self.buffer_length):
                    res.append(mymodel3(u))
                # plt.scatter(np.linspace(1,buffer_length,buffer_length), p, color='red')
                # plt.plot(np.linspace(1,buffer_length,buffer_length), res, color='green')
                pred_reg = (
                    mymodel3(self.buffer_length + self.skip_predictions)
                    + prediction[i][j] * self.weight_prediction_impact_on_regression
                ) / (self.weight_prediction_impact_on_regression + 1)

                mymodel = np.poly1d(
                    np.polyfit(
                        np.linspace(1, self.buffer_length - 1, self.buffer_length - 1),
                        last_joint_avgs,
                        1,
                    )
                )
                avg_reg = (
                    mymodel(self.buffer_length + self.skip_predictions)
                    + prediction[i][j] * self.weight_prediction_impact_on_regression
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

                self._last_avg[-1][i][j] = ges_pred

        if self.additional_filter:
            new = self._last_avg[-2] + (
                _normalize_numpy(np.sqrt(np.square(self._last_avg[-1] - self._last_avg[-2]))) / 3
            ) * (self._last_avg[-1] - self._last_avg[-2])
            self._last_avg[-1] = new

            self._last_val.pop(0)
            self._last_avg.pop(0)

            return new

        self._last_val.pop(0)
        self._last_avg.pop(0)

        return self._last_avg[-1]

    def torch_enhanced_filter(self, prediction: torch.Tensor) -> np.ndarray:
        prediction = torch.from_numpy(prediction.T)

        if self._last_avg is None and self._last_val is None:
            self._last_avg = prediction[None, ...]
            self._last_val = prediction[None, ...]
        else:
            self._last_avg = _append_to_tensor(self._last_avg, prediction)
            self._last_val = _append_to_tensor(self._last_val, prediction)

        xs = np.arange(0, self.buffer_length, 1)

        if self._last_avg.shape[0] <= 1:
            return prediction.numpy()

        if self.only_additional_filter:
            pred_last_avg = prediction - self._last_avg[-2]
            new = (
                self._last_avg[-2]
                + (_normalize_torch(pred_last_avg.abs()) / self.weight_additional_filter) * pred_last_avg
            )
            self._last_avg[-1] = new

            if self._last_avg.shape[0] > self.buffer_length:
                self._last_avg = _pop_first_in_tensor(self._last_avg)

            return new.numpy()

        if 1 < self._last_avg.shape[0] < self.buffer_length:
            pred_last_avg = prediction - self._last_avg[-2]
            new = (
                self._last_avg[-2]
                + (_normalize_torch(pred_last_avg.abs()) / self.weight_additional_filter) * pred_last_avg
            )
            self._last_avg[-1] = new

            return new.numpy()

        last_val_models = np.polyfit(xs, self._last_val.reshape(self.buffer_length, -1), 1).T
        last_avg_models = np.polyfit(xs, self._last_avg.reshape(self.buffer_length, -1), 1).T

        last_val_res = np.array(list(map(lambda x: np.poly1d(x)(xs), last_val_models)))
        last_avg_res = np.array(list(map(lambda x: np.poly1d(x)(xs), last_avg_models)))

        pred_reg = (
            torch.from_numpy(
                np.array(
                    list(map(lambda x: np.poly1d(x)(self.buffer_length + self.skip_predictions), last_val_models))
                ).reshape(prediction.shape)
            )
            + prediction * self.weight_prediction_impact_on_regression
        ) / (self.weight_prediction_impact_on_regression + 1)

        avg_reg = (
            torch.from_numpy(
                np.array(
                    list(map(lambda x: np.poly1d(x)(self.buffer_length + self.skip_predictions), last_avg_models))
                ).reshape(prediction.shape)
            )
            + prediction * self.weight_prediction_impact_on_regression
        ) / (self.weight_prediction_impact_on_regression + 1)

        crosscoeffs = torch.tensor(
            np.array(
                list(
                    map(
                        lambda x, y: np.nan_to_num(np.abs(np.corrcoef(x, y)[0, 1]), nan=1),
                        last_avg_res,
                        last_val_res,
                    )
                )
            ).reshape(prediction.shape)
        )

        self._last_avg[-1] = (avg_reg * crosscoeffs + pred_reg) / (crosscoeffs + 1)

        if self.additional_filter:
            last_avg_1_minus_2 = self._last_avg[-1] - self._last_avg[-2]

            new = self._last_avg[-2] + (_normalize_torch(last_avg_1_minus_2.abs()) / 3) * last_avg_1_minus_2
            self._last_avg[-1] = new

            self._last_avg = _pop_first_in_tensor(self._last_avg)
            self._last_val = _pop_first_in_tensor(self._last_val)

            return new.numpy()

        self._last_avg = _pop_first_in_tensor(self._last_avg)
        self._last_val = _pop_first_in_tensor(self._last_val)

        return self._last_avg[-1].numpy()

    def torch_enhanced_filter_v2(self, prediction: torch.Tensor) -> np.ndarray:
        prediction = prediction

        if self._last_avg is None and self._last_val is None:
            self._last_avg = [prediction]
            self._last_val = [prediction]
        else:
            self._last_avg.append(prediction)
            self._last_val.append(prediction)

        if len(self._last_avg) <= 1:
            return prediction.cpu().numpy()

        if self.only_additional_filter:
            pred_last_avg = prediction - self._last_avg[-2]
            new = (
                self._last_avg[-2]
                + (_normalize_torch(pred_last_avg.abs()) / self.weight_additional_filter) * pred_last_avg
            )
            self._last_avg[-1] = new

            if len(self._last_avg) > self.buffer_length:
                self._last_avg.pop(0)

            return new.cpu().numpy()

        if 1 < len(self._last_avg) < self.buffer_length:
            pred_last_avg = prediction - self._last_avg[-2]
            new = (
                self._last_avg[-2]
                + (_normalize_torch(pred_last_avg.abs()) / self.weight_additional_filter) * pred_last_avg
            )
            self._last_avg[-1] = new

            return new.cpu().numpy()

        last_val_res, last_avg_res, pred_reg, avg_reg = _compute_torch_optimized_michael_filter(
            self._A,
            self._xs,
            prediction,
            torch.stack(self._last_val),
            torch.stack(self._last_avg),
            self.buffer_length,
            self.skip_predictions,
            self.weight_prediction_impact_on_regression,
        )

        crosscoeffs = torch.nan_to_num(
            torch.abs(self._pearson(last_val_res, last_avg_res).view(prediction.shape)),
            nan=1,
        )

        self._last_avg[-1] = (avg_reg * crosscoeffs + pred_reg) / (crosscoeffs + 1)

        if self.additional_filter:
            last_avg_1_minus_2 = self._last_avg[-1] - self._last_avg[-2]

            new = self._last_avg[-2] + (_normalize_torch(last_avg_1_minus_2.abs()) / 3) * last_avg_1_minus_2
            self._last_avg[-1] = new

            self._last_avg.pop(0)
            self._last_val.pop(0)

            return new.cpu().numpy()

        self._last_avg.pop(0)
        self._last_val.pop(0)

        return self._last_avg[-1].cpu().numpy()

    def numba_enhanced_filter(self, prediction: torch.FloatTensor) -> np.ndarray:
        prediction = prediction.T

        if self._last_avg is None and self._last_val is None:
            self._last_avg = NumbaList([prediction])
            self._last_val = NumbaList([prediction])
        else:
            self._last_avg.append(prediction)
            self._last_val.append(prediction)

        if len(self._last_avg) <= 1:
            return prediction

        if self.only_additional_filter:
            new = _additional_filter(
                last_avg=self._last_avg, prediction=prediction, weight_additional_filter=self.weight_additional_filter
            )

            if len(self._last_avg) > self.buffer_length:
                self._last_avg.pop(0)

            return new

        if 1 < len(self._last_avg) < self.buffer_length:
            return _additional_filter(
                last_avg=self._last_avg, prediction=prediction, weight_additional_filter=self.weight_additional_filter
            )

        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                last_joint_preds = []
                last_joint_avgs = []
                for k in range(self.buffer_length - 1):
                    last_joint_preds.append(self._last_val[k][i][j])
                for k in range(self.buffer_length - 1):
                    last_joint_avgs.append(self._last_avg[k][i][j])

                mymodel3 = fit_poly(
                    np.linspace(1, self.buffer_length - 1, self.buffer_length - 1),
                    np.array(last_joint_preds),
                    1,
                )

                res = NumbaList()
                for u in range(self.buffer_length):
                    res.append(eval_polynomial(mymodel3, u))
                # plt.scatter(np.linspace(1,buffer_length,buffer_length), p, color='red')
                # plt.plot(np.linspace(1,buffer_length,buffer_length), res, color='green')
                pred_reg = (
                    eval_polynomial(mymodel3, self.buffer_length + self.skip_predictions)
                    + prediction[i][j] * self.weight_prediction_impact_on_regression
                ) / (self.weight_prediction_impact_on_regression + 1)

                mymodel = fit_poly(
                    np.linspace(1, self.buffer_length - 1, self.buffer_length - 1),
                    np.array(last_joint_avgs),
                    1,
                )

                avg_reg = (
                    eval_polynomial(mymodel, self.buffer_length + self.skip_predictions)
                    + prediction[i][j] * self.weight_prediction_impact_on_regression
                ) / (self.weight_prediction_impact_on_regression + 1)
                res2 = []
                for u in range(self.buffer_length):
                    res2.append(eval_polynomial(mymodel, u))
                # plt.plot(np.linspace(1, buffer_length, buffer_length), res2, color='blue')
                # plt.scatter(np.linspace(1, buffer_length, buffer_length), q, color='black')
                # plt.show()

                crozzcoeff = np.abs(np.corrcoef(res2, res)[0, 1])
                if np.isnan(crozzcoeff):
                    crozzcoeff = 1

                ges_pred = (avg_reg * crozzcoeff + pred_reg) / (crozzcoeff + 1)
                # ges_pred = ((mymodel3(buffer_length) + mymodel(buffer_length) + prediction[i][j]*var2) / (2 + var2))

                self._last_avg[-1][i][j] = ges_pred

        if self.additional_filter:
            new = self._last_avg[-2] + (
                _normalize_numpy(np.sqrt(np.square(self._last_avg[-1] - self._last_avg[-2]))) / 3
            ) * (self._last_avg[-1] - self._last_avg[-2])
            self._last_avg[-1] = new

            self._last_val.pop(0)
            self._last_avg.pop(0)

            return new

        self._last_val.pop(0)
        self._last_avg.pop(0)

        return self._last_avg[-1]


@torch.jit.script
def _compute_torch_optimized_michael_filter(
    A: torch.Tensor,
    xs: torch.Tensor,
    prediction: torch.Tensor,
    _last_val: torch.Tensor,
    _last_avg: torch.Tensor,
    buffer_length: int,
    skip_prediction: int,
    weight_prediction_impact_on_regression: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute torch optimized michael filter

    Parameters
    ----------
    A : torch.Tensor
        The A matrix used for Ax = b
    xs : torch.Tensor
        The xs vector used for Ax = b
    prediction : torch.Tensor
        The prediction
    _last_val : torch.Tensor
        The last values
    _last_avg : torch.Tensor
        The last averages
    buffer_length : int
        The buffer length
    skip_prediction : int
        The skip prediction
    weight_prediction_impact_on_regression : float
        The weight prediction impact on regression

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The new last values, the new last averages, the new prediction and the new average
    """

    # Do Ax = b to basically get an 1 deg polynomial fit. The highest order coefficient is the last one in the vector!
    last_val_models = torch.linalg.lstsq(A, _last_val.view(buffer_length, -1).T, rcond=None)[0]
    last_avg_models = torch.linalg.lstsq(A, _last_avg.view(buffer_length, -1).T, rcond=None)[0]

    # Use Horner's method (yes ... I also didn't think I'll use it after my 1st semester ... life do be like that)
    # to evaluate the polynomial. Such wow. Much performance. Very fast.
    last_val_res = last_val_models[:, 1].expand(len(xs), -1) * xs[:, None] + last_val_models[:, 0].expand(len(xs), -1)
    last_avg_res = last_avg_models[:, 1].expand(len(xs), -1) * xs[:, None] + last_avg_models[:, 0].expand(len(xs), -1)

    # Do some regression stuff defined in the original implementation by Michael.
    pred_reg = (
        (last_val_models[:, 1] * (buffer_length + skip_prediction) + last_val_models[:, 0]).view(prediction.shape)
        + prediction * weight_prediction_impact_on_regression
    ) / (weight_prediction_impact_on_regression + 1)

    avg_reg = (
        (last_avg_models[:, 1] * (buffer_length + skip_prediction) + last_avg_models[:, 0]).view(prediction.shape)
        + prediction * weight_prediction_impact_on_regression
    ) / (weight_prediction_impact_on_regression + 1)

    return last_val_res, last_avg_res, pred_reg, avg_reg
