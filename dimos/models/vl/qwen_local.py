from dimos.models.vl.qwen import QwenVlModel, QwenVlModelConfig


class QwenLocalVlModelConfig(QwenVlModelConfig):
    """Configuration for a locally-deployed Qwen VL model.

    Expects an OpenAI-compatible server (vLLM, SGLang, or Ollama)
    running on the specified base_url.

    Example with vLLM::

        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-VL-3B-Instruct \
            --port 8000

    Then configure::

        QwenLocalVlModel(base_url="http://<host>:8000/v1", model_name="Qwen/Qwen2.5-VL-3B-Instruct")
    """

    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"


class QwenLocalVlModel(QwenVlModel):
    """Qwen VL model backed by a local OpenAI-compatible server.

    Drop-in replacement for QwenVlModel — same interface, zero cloud
    API keys required. Works with vLLM, SGLang, or Ollama as the backend.
    """

    default_config = QwenLocalVlModelConfig

    @property
    def _client(self):  # type: ignore[override]
        if "_client_instance" not in self.__dict__:
            from openai import OpenAI

            self.__dict__["_client_instance"] = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self.__dict__["_client_instance"]

    def stop(self) -> None:
        if "_client_instance" in self.__dict__:
            del self.__dict__["_client_instance"]


__all__ = ["QwenLocalVlModel", "QwenLocalVlModelConfig"]
