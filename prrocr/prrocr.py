"""OCR related modeling class"""

from typing import Optional
from dataclasses import dataclass
from prrocr.tasks import download_or_load
import torch
# from prrocr.tasks.utils.base import PororoFactoryBase, PororoSimpleBase

@dataclass
class TaskConfig:
    task: str
    lang: str
    n_model: str

class PororoOCR:
    """
    Recognize optical characters in image file
    Currently support Korean language

    English + Korean (`brainocr`)

        - dataset: Internal data + AI hub Font Image dataset
        - metric: TBU
        - ref: https://www.aihub.or.kr/aidata/133

    Examples:
        >>> ocr = PororoOCR(lang="ko")
        >>> ocr(IMAGE_PATH)
        ["사이렌'(' 신마'", "내가 말했잖아 속지열라고 이 손을 잡는 너는 위협해질 거라고"]

        >>> ocr = PororoOCR(lang="ko")
        >>> ocr(IMAGE_PATH, detail=True)
        {
            'description': ["사이렌'(' 신마', "내가 말했잖아 속지열라고 이 손을 잡는 너는 위협해질 거라고"],
            'bounding_poly': [
                {
                    'description': "사이렌'(' 신마'",
                    'vertices': [
                        {'x': 93, 'y': 7},
                        {'x': 164, 'y': 7},
                        {'x': 164, 'y': 21},
                        {'x': 93, 'y': 21}
                    ]
                },
                {
                    'description': "내가 말했잖아 속지열라고 이 손을 잡는 너는 위협해질 거라고",
                    'vertices': [
                        {'x': 0, 'y': 30},
                        {'x': 259, 'y': 30},
                        {'x': 259, 'y': 194},
                        {'x': 0, 'y': 194}]}
                    ]
                }
        }
    """

    def __init__(
            self,
            lang: str,
            model: Optional[str]=None
    ):
        self._available_langs = self.get_available_langs()
        self._available_models = self.get_available_models()
        self._model2lang = {
            v: k for k, vs in self._available_models.items() for v in vs
        }
        # Set default language as very first supported language
        assert (
            lang in self._available_langs
        ), f"Following langs are supported for this task: {self._available_langs}"

        if lang is None:
            lang = self._available_langs[0]

        # Change language option if model is defined by user
        if model is not None:
            lang = self._model2lang[model]

        # Set default model
        if model is None:
            model = self.get_default_model(lang)

        # yapf: disable
        assert (model in self._available_models[lang]), f"{model} is NOT supported for {lang}"
        # yapf: enable

        self._model = model
        self.detect_model="craft"
        self.ocr_opt = "ocr-opt"
        self.config = TaskConfig("ocr", lang, model)


        # Get device information from torch API
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate task-specific pipeline module, if possible
        self.model = self.load(device)


    def __call__(self, text, **kwargs):
        return self.predict(text, **kwargs)

    @staticmethod
    def get_available_langs():
        return ["en", "ko"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["brainocr"],
            "ko": ["brainocr"],
        }

    def get_default_model(self, lang: str) -> str:
        return self._available_models[lang][0]

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.n_model == "brainocr":
            from prrocr.models.brainOCR import brainocr

            if self.config.lang not in self.get_available_langs():
                raise ValueError(
                    f"Unsupported Language : {self.config.lang}",
                    'Support Languages : ["en", "ko"]',
                )

            det_model_path = download_or_load(
                f"misc/{self.detect_model}.pt",
                self.config.lang,
            )
            rec_model_path = download_or_load(
                f"misc/{self.config.n_model}.pt",
                self.config.lang,
            )
            opt_fp = download_or_load(
                f"misc/{self.ocr_opt}.txt",
                self.config.lang,
            )
            model = brainocr.Reader(
                self.config.lang,
                det_model_ckpt_fp=det_model_path,
                rec_model_ckpt_fp=rec_model_path,
                opt_fp=opt_fp,
                device=device,
            )
            model.detector.to(device)
            model.recognizer.to(device)
            return model

    def _postprocess(self, ocr_results, detail: bool = False):
        """
        Post-process for OCR result

        Args:
            ocr_results (list): list contains result of OCR
            detail (bool): if True, returned to include details. (bounding poly, vertices, etc)

        """
        sorted_ocr_results = sorted(
            ocr_results,
            key=lambda x: (
                x[0][0][1],
                x[0][0][0],
            ),
        )

        if not detail:
            return [
                sorted_ocr_results[i][-1]
                for i in range(len(sorted_ocr_results))
            ]

        result_dict = {
            "description": list(),
            "bounding_poly": list(),
        }

        for ocr_result in sorted_ocr_results:
            vertices = list()

            for vertice in ocr_result[0]:
                vertices.append({
                    "x": vertice[0],
                    "y": vertice[1],
                })

            result_dict["description"].append(ocr_result[1])
            result_dict["bounding_poly"].append({
                "description": ocr_result[1],
                "vertices": vertices
            })

        return result_dict

    def predict(self, image_path: str, **kwargs):
        """
        Conduct Optical Character Recognition (OCR)

        Args:
            image_path (str): the image file path
            detail (bool): if True, returned to include details. (bounding poly, vertices, etc)

        """
        detail = kwargs.get("detail", False)
        return self._postprocess(
            self.model(
                image_path,
                skip_details=False,
                batch_size=1,
                paragraph=True,
            ),
            detail,
        )
