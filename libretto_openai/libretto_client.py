from itertools import tee
import json
import logging
import os
import uuid

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Union,
    Optional,
    cast,
    overload,
)
from typing_extensions import Literal

import httpx
import openai
from openai import resources
from openai._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from openai._utils import required_args
from openai.types.completion import Completion
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)

from .template import TemplateChat, TemplateString
from .patch import (
    LibrettoCreateParamDict,
    LibrettoCreateParams,
    event_session,
)
from .pii import Redactor


logger = logging.getLogger(__name__)


class LibrettoConfig(NamedTuple):
    api_key: str | None = None
    prompt_template_name: str | None = None
    chat_id: str | None = None
    allow_unnamed_prompts: bool = False
    redact_pii: bool = False


class LibrettoCompletionsBaseMixin:
    def __init__(self, config: LibrettoConfig) -> None:
        self.config = config
        self.pii_redactor = Redactor() if config.redact_pii else None

    def _original_create(self, *args, **kwargs):
        raise NotImplementedError()

    def _get_result(self, response):
        raise NotImplementedError()

    def _create(self, libretto: LibrettoCreateParamDict | None, *original_args, **original_kwargs):
        libretto = self._prepare_libretto(libretto)

        if libretto["project_key"] is None and libretto["api_key"] is None:
            return self._original_create(*original_args, **original_kwargs)

        if libretto["prompt_template_name"] is None and not self.config.allow_unnamed_prompts:
            return self._original_create(*original_args, **original_kwargs)

        model_params = self._build_model_params(**original_kwargs)

        # Redact PII from template parameters if configured to do so
        self._redact_template_params(libretto["template_params"])

        with event_session(
            project_key=libretto["project_key"],
            api_key=libretto["api_key"],
            prompt_template_name=libretto["prompt_template_name"],
            model_params=model_params,
            prompt_template_text=libretto["template_text"],
            prompt_template_chat=libretto["template_chat"],
            chat_id=libretto["chat_id"],
            prompt_template_params=libretto["template_params"],
            prompt_event_id=libretto["event_id"],
            parent_event_id=libretto["parent_event_id"],
            feedback_key=libretto["feedback_key"],
        ) as complete_event:
            response = self._original_create(*original_args, **original_kwargs)
            return_response, event_response = self._get_result(response)

            # Can only do this for non-streamed responses right now
            if not original_kwargs.get("stream"):
                setattr(return_response, "libretto_feedback_key", libretto["feedback_key"])

            # Redact PII before recording the event
            event_response = self._redact_response(event_response)

            complete_event(event_response)

            return return_response

    def _prepare_libretto(
        self, libretto: LibrettoCreateParamDict | None = None
    ) -> LibrettoCreateParamDict:
        if libretto is None:
            libretto = LibrettoCreateParams()
        else:
            # Don't mutate the input dict
            libretto = libretto.copy()

        libretto["prompt_template_name"] = (
            libretto["prompt_template_name"]
            or self.config.prompt_template_name
            or os.environ.get("LIBRETTO_TEMPLATE_NAME")
            or libretto["api_name"]  # legacy
        )
        libretto["api_key"] = (
            libretto["api_key"] or self.config.api_key or os.environ.get("LIBRETTO_API_KEY")
        )
        libretto["chat_id"] = (
            libretto["chat_id"] or self.config.chat_id or os.environ.get("LIBRETTO_CHAT_ID")
        )
        libretto["project_key"] = libretto["project_key"] or os.environ.get("LIBRETTO_PROJECT_KEY")
        libretto["feedback_key"] = libretto["feedback_key"] or str(uuid.uuid4())

        return libretto

    def _build_model_params(self, **original_kwargs) -> Dict[str, Any]:
        model_params = {
            "modelProvider": "openai",
        }
        for k, v in original_kwargs.items():
            if v == NOT_GIVEN:
                continue
            model_params[k] = v
        return model_params

    def _redact_template_params(self, template_params) -> None:
        if not template_params:
            return
        if not self.pii_redactor:
            return

        for name, param in template_params.items():
            try:
                template_params[name] = self.pii_redactor.redact(param)
            except Exception as e:
                logger.warning(
                    "Failed to redact PII from parameter: key=%s, value=%s, error=%s",
                    name,
                    param,
                    e,
                )

    def _redact_response(self, event_response: str | None) -> str:
        if not event_response:
            return ""
        if not self.pii_redactor:
            return ""

        try:
            return self.pii_redactor.redact_text(event_response)
        except Exception as e:
            logger.warning(
                "Failed to redact PII from response: error=%s",
                e,
            )
            return event_response


class LibrettoCompletionsMixin(LibrettoCompletionsBaseMixin):
    def _original_create(self, *args, **kwargs):
        raise NotImplementedError()

    def _prepare_libretto(
        self,
        libretto: LibrettoCreateParamDict | None = None,
        **original_kwargs,
    ) -> LibrettoCreateParamDict:
        libretto = super()._prepare_libretto(libretto=libretto, **original_kwargs)

        prompt = original_kwargs.get("prompt", "")
        if isinstance(prompt, TemplateString):
            libretto["template_text"] = prompt.template
            if libretto["template_params"] is None:
                libretto["template_params"] = {}
            libretto["template_params"].update(prompt.params)
        else:
            libretto["template_text"] = prompt

        return libretto

    def _build_model_params(self, **original_kwargs) -> Dict[str, Any]:
        model_params = super()._build_model_params(**original_kwargs)
        model_params["modelType"] = "completion"
        del model_params["prompt"]
        return model_params

    def _get_result(self, response: Completion | Iterable[Completion]):
        if isinstance(response, Completion):
            return response, response.choices[0].text

        original, consumable = tee(response)
        accumulated = []
        for response_chunk in consumable:
            accumulated.append(response_chunk.choices[0].text)
        return (original, "".join(accumulated))


class LibrettoChatCompletionsMixin(LibrettoCompletionsBaseMixin):
    def _original_create(self, *args, **kwargs):
        raise NotImplementedError()

    def _prepare_libretto(
        self,
        libretto: LibrettoCreateParamDict | None = None,
        **original_kwargs,
    ) -> LibrettoCreateParamDict:
        libretto = super()._prepare_libretto(libretto=libretto, **original_kwargs)

        messages = original_kwargs.get("messages")
        if hasattr(messages, "template"):
            libretto["template_chat"] = cast(TemplateChat, messages).template
        if hasattr(messages, "params"):
            if libretto["template_params"] is None:
                libretto["template_params"] = {}
            libretto["template_params"].update(cast(TemplateChat, messages).params)

        return libretto

    def _build_model_params(self, **original_kwargs) -> Dict[str, Any]:
        model_params = super()._build_model_params(**original_kwargs)
        model_params["modelType"] = "chat"
        del model_params["messages"]
        return model_params

    def _get_result(self, response: Iterable[ChatCompletionChunk] | ChatCompletion):
        if not isinstance(response, ChatCompletion):
            return self._get_stream_result(response)

        message = response.choices[0].message
        if message.function_call:
            return response, json.dumps({"function_call": message.function_call})
        return response, message.content

    def _get_stream_result(self, responses: Iterable[ChatCompletionChunk]):
        (original_response, consumable_response) = tee(responses)
        accumulated = []
        for response in consumable_response:
            if response.choices[0].delta.content is not None:
                accumulated.append(response.choices[0].delta.content)
            if response.choices[0].delta.function_call is not None:
                logger.warning("Streaming a function_call response is not supported yet.")
        return (original_response, "".join(accumulated))


class LibrettoCompletions(resources.Completions, LibrettoCompletionsMixin):
    def __init__(self, client: openai.Client, config: LibrettoConfig) -> None:
        super().__init__(client)
        LibrettoCompletionsMixin.__init__(self, config)

    @overload
    def create(
        self,
        *,
        model: Union[
            str,
            Literal[
                "babbage-002",
                "davinci-002",
                "gpt-3.5-turbo-instruct",
                "text-davinci-003",
                "text-davinci-002",
                "text-davinci-001",
                "code-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
        ],
        prompt: Union[str, List[str], List[int], List[List[int]], None, TemplateString],
        best_of: Optional[int] | NotGiven = NOT_GIVEN,
        echo: Optional[bool] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        libretto: LibrettoCreateParamDict | None = None,
    ) -> Completion:
        """
        Creates a completion for the provided prompt and parameters.

        Args:
          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          prompt: The prompt(s) to generate completions for, encoded as a string, array of
              strings, array of tokens, or array of token arrays.

              Note that <|endoftext|> is the document separator that the model sees during
              training, so if a prompt is not specified the model will generate as if from the
              beginning of a new document.

          best_of: Generates `best_of` completions server-side and returns the "best" (the one with
              the highest log probability per token). Results cannot be streamed.

              When used with `n`, `best_of` controls the number of candidate completions and
              `n` specifies how many to return – `best_of` must be greater than `n`.

              **Note:** Because this parameter generates many completions, it can quickly
              consume your token quota. Use carefully and ensure that you have reasonable
              settings for `max_tokens` and `stop`.

          echo: Echo back the prompt in addition to the completion

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the GPT
              tokenizer) to an associated bias value from -100 to 100. You can use this
              [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to
              convert text to token IDs. Mathematically, the bias is added to the logits
              generated by the model prior to sampling. The exact effect will vary per model,
              but values between -1 and 1 should decrease or increase likelihood of selection;
              values like -100 or 100 should result in a ban or exclusive selection of the
              relevant token.

              As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token
              from being generated.

          logprobs: Include the log probabilities on the `logprobs` most likely tokens, as well the
              chosen tokens. For example, if `logprobs` is 5, the API will return a list of
              the 5 most likely tokens. The API will always return the `logprob` of the
              sampled token, so there may be up to `logprobs+1` elements in the response.

              The maximum value for `logprobs` is 5.

          max_tokens: The maximum number of [tokens](/tokenizer) to generate in the completion.

              The token count of your prompt plus `max_tokens` cannot exceed the model's
              context length.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          n: How many completions to generate for each prompt.

              **Note:** Because this parameter generates many completions, it can quickly
              consume your token quota. Use carefully and ensure that you have reasonable
              settings for `max_tokens` and `stop`.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          seed: If specified, our system will make a best effort to sample deterministically,
              such that repeated requests with the same `seed` and parameters should return
              the same result.

              Determinism is not guaranteed, and you should refer to the `system_fingerprint`
              response parameter to monitor changes in the backend.

          stop: Up to 4 sequences where the API will stop generating further tokens. The
              returned text will not contain the stop sequence.

          stream: Whether to stream back partial progress. If set, tokens will be sent as
              data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available, with the stream terminated by a `data: [DONE]`
              message.
              [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          suffix: The suffix that comes after a completion of inserted text.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def create(
        self,
        *,
        model: Union[
            str,
            Literal[
                "babbage-002",
                "davinci-002",
                "gpt-3.5-turbo-instruct",
                "text-davinci-003",
                "text-davinci-002",
                "text-davinci-001",
                "code-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
        ],
        prompt: Union[str, List[str], List[int], List[List[int]], None, TemplateString],
        stream: Literal[True],
        best_of: Optional[int] | NotGiven = NOT_GIVEN,
        echo: Optional[bool] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        libretto: LibrettoCreateParamDict | None = None,
    ) -> Iterable[Completion]:
        """
        Creates a completion for the provided prompt and parameters.

        Args:
          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          prompt: The prompt(s) to generate completions for, encoded as a string, array of
              strings, array of tokens, or array of token arrays.

              Note that <|endoftext|> is the document separator that the model sees during
              training, so if a prompt is not specified the model will generate as if from the
              beginning of a new document.

          stream: Whether to stream back partial progress. If set, tokens will be sent as
              data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available, with the stream terminated by a `data: [DONE]`
              message.
              [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          best_of: Generates `best_of` completions server-side and returns the "best" (the one with
              the highest log probability per token). Results cannot be streamed.

              When used with `n`, `best_of` controls the number of candidate completions and
              `n` specifies how many to return – `best_of` must be greater than `n`.

              **Note:** Because this parameter generates many completions, it can quickly
              consume your token quota. Use carefully and ensure that you have reasonable
              settings for `max_tokens` and `stop`.

          echo: Echo back the prompt in addition to the completion

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the GPT
              tokenizer) to an associated bias value from -100 to 100. You can use this
              [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to
              convert text to token IDs. Mathematically, the bias is added to the logits
              generated by the model prior to sampling. The exact effect will vary per model,
              but values between -1 and 1 should decrease or increase likelihood of selection;
              values like -100 or 100 should result in a ban or exclusive selection of the
              relevant token.

              As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token
              from being generated.

          logprobs: Include the log probabilities on the `logprobs` most likely tokens, as well the
              chosen tokens. For example, if `logprobs` is 5, the API will return a list of
              the 5 most likely tokens. The API will always return the `logprob` of the
              sampled token, so there may be up to `logprobs+1` elements in the response.

              The maximum value for `logprobs` is 5.

          max_tokens: The maximum number of [tokens](/tokenizer) to generate in the completion.

              The token count of your prompt plus `max_tokens` cannot exceed the model's
              context length.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          n: How many completions to generate for each prompt.

              **Note:** Because this parameter generates many completions, it can quickly
              consume your token quota. Use carefully and ensure that you have reasonable
              settings for `max_tokens` and `stop`.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          seed: If specified, our system will make a best effort to sample deterministically,
              such that repeated requests with the same `seed` and parameters should return
              the same result.

              Determinism is not guaranteed, and you should refer to the `system_fingerprint`
              response parameter to monitor changes in the backend.

          stop: Up to 4 sequences where the API will stop generating further tokens. The
              returned text will not contain the stop sequence.

          suffix: The suffix that comes after a completion of inserted text.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def create(
        self,
        *,
        model: Union[
            str,
            Literal[
                "babbage-002",
                "davinci-002",
                "gpt-3.5-turbo-instruct",
                "text-davinci-003",
                "text-davinci-002",
                "text-davinci-001",
                "code-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
        ],
        prompt: Union[str, List[str], List[int], List[List[int]], None, TemplateString],
        stream: bool,
        best_of: Optional[int] | NotGiven = NOT_GIVEN,
        echo: Optional[bool] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        libretto: LibrettoCreateParamDict | None = None,
    ) -> Completion | Iterable[Completion]:
        """
        Creates a completion for the provided prompt and parameters.

        Args:
          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          prompt: The prompt(s) to generate completions for, encoded as a string, array of
              strings, array of tokens, or array of token arrays.

              Note that <|endoftext|> is the document separator that the model sees during
              training, so if a prompt is not specified the model will generate as if from the
              beginning of a new document.

          stream: Whether to stream back partial progress. If set, tokens will be sent as
              data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available, with the stream terminated by a `data: [DONE]`
              message.
              [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          best_of: Generates `best_of` completions server-side and returns the "best" (the one with
              the highest log probability per token). Results cannot be streamed.

              When used with `n`, `best_of` controls the number of candidate completions and
              `n` specifies how many to return – `best_of` must be greater than `n`.

              **Note:** Because this parameter generates many completions, it can quickly
              consume your token quota. Use carefully and ensure that you have reasonable
              settings for `max_tokens` and `stop`.

          echo: Echo back the prompt in addition to the completion

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the GPT
              tokenizer) to an associated bias value from -100 to 100. You can use this
              [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to
              convert text to token IDs. Mathematically, the bias is added to the logits
              generated by the model prior to sampling. The exact effect will vary per model,
              but values between -1 and 1 should decrease or increase likelihood of selection;
              values like -100 or 100 should result in a ban or exclusive selection of the
              relevant token.

              As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token
              from being generated.

          logprobs: Include the log probabilities on the `logprobs` most likely tokens, as well the
              chosen tokens. For example, if `logprobs` is 5, the API will return a list of
              the 5 most likely tokens. The API will always return the `logprob` of the
              sampled token, so there may be up to `logprobs+1` elements in the response.

              The maximum value for `logprobs` is 5.

          max_tokens: The maximum number of [tokens](/tokenizer) to generate in the completion.

              The token count of your prompt plus `max_tokens` cannot exceed the model's
              context length.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          n: How many completions to generate for each prompt.

              **Note:** Because this parameter generates many completions, it can quickly
              consume your token quota. Use carefully and ensure that you have reasonable
              settings for `max_tokens` and `stop`.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          seed: If specified, our system will make a best effort to sample deterministically,
              such that repeated requests with the same `seed` and parameters should return
              the same result.

              Determinism is not guaranteed, and you should refer to the `system_fingerprint`
              response parameter to monitor changes in the backend.

          stop: Up to 4 sequences where the API will stop generating further tokens. The
              returned text will not contain the stop sequence.

          suffix: The suffix that comes after a completion of inserted text.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @required_args(["model", "prompt"], ["model", "prompt", "stream"])
    def create(
        self,
        *args,
        model: Union[
            str,
            Literal[
                "babbage-002",
                "davinci-002",
                "gpt-3.5-turbo-instruct",
                "text-davinci-003",
                "text-davinci-002",
                "text-davinci-001",
                "code-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
        ],
        prompt: Union[str, List[str], List[int], List[List[int]], None, TemplateString],
        best_of: Optional[int] | NotGiven = NOT_GIVEN,
        echo: Optional[bool] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        libretto: LibrettoCreateParamDict | None = None,
    ) -> Completion | Iterable[Completion]:
        original_kwargs = dict(
            model=model,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            stream=stream,
            suffix=suffix,
            temperature=temperature,
            top_p=top_p,
            user=user,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
        return self._create(libretto, *args, **original_kwargs)

    def _original_create(self, *args, **kwargs):
        return super().create(*args, **kwargs)


class LibrettoChatCompletions(resources.chat.Completions, LibrettoChatCompletionsMixin):
    def __init__(self, client: openai.Client, config: LibrettoConfig) -> None:
        super().__init__(client)
        LibrettoChatCompletionsMixin.__init__(self, config)

    @overload
    def create(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: Union[
            str,
            Literal[
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-4-32k-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
            ],
        ],
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: List[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        libretto: LibrettoCreateParamDict | None = None,
    ) -> ChatCompletion:
        """
        Creates a model response for the given chat conversation.

        Args:
          messages: A list of messages comprising the conversation so far.
              [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).

          model: ID of the model to use. See the
              [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility)
              table for details on which models work with the Chat API.

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          function_call: Deprecated in favor of `tool_choice`.

              Controls which (if any) function is called by the model. `none` means the model
              will not call a function and instead generates a message. `auto` means the model
              can pick between generating a message or calling a function. Specifying a
              particular function via `{"name": "my_function"}` forces the model to call that
              function.

              `none` is the default when no functions are present. `auto`` is the default if
              functions are present.

          functions: Deprecated in favor of `tools`.

              A list of functions the model may generate JSON inputs for.

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the
              tokenizer) to an associated bias value from -100 to 100. Mathematically, the
              bias is added to the logits generated by the model prior to sampling. The exact
              effect will vary per model, but values between -1 and 1 should decrease or
              increase likelihood of selection; values like -100 or 100 should result in a ban
              or exclusive selection of the relevant token.

          max_tokens: The maximum number of [tokens](/tokenizer) to generate in the chat completion.

              The total length of input tokens and generated tokens is limited by the model's
              context length.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          n: How many chat completion choices to generate for each input message.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          response_format: An object specifying the format that the model must output. Used to enable JSON
              mode.

          seed: This feature is in Beta. If specified, our system will make a best effort to
              sample deterministically, such that repeated requests with the same `seed` and
              parameters should return the same result. Determinism is not guaranteed, and you
              should refer to the `system_fingerprint` response parameter to monitor changes
              in the backend.

          stop: Up to 4 sequences where the API will stop generating further tokens.

          stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available, with the stream terminated by a `data: [DONE]`
              message.
              [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          tool_choice: Controls which (if any) function is called by the model. `none` means the model
              will not call a function and instead generates a message. `auto` means the model
              can pick between generating a message or calling a function. Specifying a
              particular function via
              `{"type: "function", "function": {"name": "my_function"}}` forces the model to
              call that function.

              `none` is the default when no functions are present. `auto` is the default if
              functions are present.

          tools: A list of tools the model may call. Currently, only functions are supported as a
              tool. Use this to provide a list of functions the model may generate JSON inputs
              for.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def create(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: Union[
            str,
            Literal[
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-4-32k-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
            ],
        ],
        stream: Literal[True],
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: List[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        libretto: LibrettoCreateParamDict | None = None,
    ) -> Iterable[ChatCompletionChunk]:
        """
        Creates a model response for the given chat conversation.

        Args:
          messages: A list of messages comprising the conversation so far.
              [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).

          model: ID of the model to use. See the
              [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility)
              table for details on which models work with the Chat API.

          stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available, with the stream terminated by a `data: [DONE]`
              message.
              [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          function_call: Deprecated in favor of `tool_choice`.

              Controls which (if any) function is called by the model. `none` means the model
              will not call a function and instead generates a message. `auto` means the model
              can pick between generating a message or calling a function. Specifying a
              particular function via `{"name": "my_function"}` forces the model to call that
              function.

              `none` is the default when no functions are present. `auto`` is the default if
              functions are present.

          functions: Deprecated in favor of `tools`.

              A list of functions the model may generate JSON inputs for.

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the
              tokenizer) to an associated bias value from -100 to 100. Mathematically, the
              bias is added to the logits generated by the model prior to sampling. The exact
              effect will vary per model, but values between -1 and 1 should decrease or
              increase likelihood of selection; values like -100 or 100 should result in a ban
              or exclusive selection of the relevant token.

          max_tokens: The maximum number of [tokens](/tokenizer) to generate in the chat completion.

              The total length of input tokens and generated tokens is limited by the model's
              context length.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          n: How many chat completion choices to generate for each input message.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          response_format: An object specifying the format that the model must output. Used to enable JSON
              mode.

          seed: This feature is in Beta. If specified, our system will make a best effort to
              sample deterministically, such that repeated requests with the same `seed` and
              parameters should return the same result. Determinism is not guaranteed, and you
              should refer to the `system_fingerprint` response parameter to monitor changes
              in the backend.

          stop: Up to 4 sequences where the API will stop generating further tokens.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          tool_choice: Controls which (if any) function is called by the model. `none` means the model
              will not call a function and instead generates a message. `auto` means the model
              can pick between generating a message or calling a function. Specifying a
              particular function via
              `{"type: "function", "function": {"name": "my_function"}}` forces the model to
              call that function.

              `none` is the default when no functions are present. `auto` is the default if
              functions are present.

          tools: A list of tools the model may call. Currently, only functions are supported as a
              tool. Use this to provide a list of functions the model may generate JSON inputs
              for.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def create(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: Union[
            str,
            Literal[
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-4-32k-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
            ],
        ],
        stream: bool,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: List[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        libretto: LibrettoCreateParamDict | None = None,
    ) -> ChatCompletion | Iterable[ChatCompletionChunk]:
        """
        Creates a model response for the given chat conversation.

        Args:
          messages: A list of messages comprising the conversation so far.
              [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).

          model: ID of the model to use. See the
              [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility)
              table for details on which models work with the Chat API.

          stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available, with the stream terminated by a `data: [DONE]`
              message.
              [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          function_call: Deprecated in favor of `tool_choice`.

              Controls which (if any) function is called by the model. `none` means the model
              will not call a function and instead generates a message. `auto` means the model
              can pick between generating a message or calling a function. Specifying a
              particular function via `{"name": "my_function"}` forces the model to call that
              function.

              `none` is the default when no functions are present. `auto`` is the default if
              functions are present.

          functions: Deprecated in favor of `tools`.

              A list of functions the model may generate JSON inputs for.

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the
              tokenizer) to an associated bias value from -100 to 100. Mathematically, the
              bias is added to the logits generated by the model prior to sampling. The exact
              effect will vary per model, but values between -1 and 1 should decrease or
              increase likelihood of selection; values like -100 or 100 should result in a ban
              or exclusive selection of the relevant token.

          max_tokens: The maximum number of [tokens](/tokenizer) to generate in the chat completion.

              The total length of input tokens and generated tokens is limited by the model's
              context length.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          n: How many chat completion choices to generate for each input message.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)

          response_format: An object specifying the format that the model must output. Used to enable JSON
              mode.

          seed: This feature is in Beta. If specified, our system will make a best effort to
              sample deterministically, such that repeated requests with the same `seed` and
              parameters should return the same result. Determinism is not guaranteed, and you
              should refer to the `system_fingerprint` response parameter to monitor changes
              in the backend.

          stop: Up to 4 sequences where the API will stop generating further tokens.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          tool_choice: Controls which (if any) function is called by the model. `none` means the model
              will not call a function and instead generates a message. `auto` means the model
              can pick between generating a message or calling a function. Specifying a
              particular function via
              `{"type: "function", "function": {"name": "my_function"}}` forces the model to
              call that function.

              `none` is the default when no functions are present. `auto` is the default if
              functions are present.

          tools: A list of tools the model may call. Currently, only functions are supported as a
              tool. Use this to provide a list of functions the model may generate JSON inputs
              for.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @required_args(["messages", "model"], ["messages", "model", "stream"])
    def create(
        self,
        *args,
        messages: List[ChatCompletionMessageParam],
        model: Union[
            str,
            Literal[
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-4-32k-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
            ],
        ],
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: List[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        libretto: LibrettoCreateParamDict | None = None,
    ) -> ChatCompletion | Iterable[ChatCompletionChunk]:
        original_kwargs = dict(
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_p=top_p,
            user=user,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
        return self._create(libretto, *args, **original_kwargs)

    def _original_create(self, *args, **kwargs):
        return super().create(*args, **kwargs)


class LibrettoChat(resources.Chat):
    completions: LibrettoChatCompletions

    def __init__(self, client: openai.Client, config: LibrettoConfig):
        super().__init__(client)
        self.completions = LibrettoChatCompletions(client, config)


class LibrettoOpenAIClient(openai.Client):
    config: LibrettoConfig
    completions: LibrettoCompletions
    chat: LibrettoChat

    def __init__(self, *args, libretto: LibrettoConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = libretto or LibrettoConfig()
        self.completions = LibrettoCompletions(self, self.config)
        self.chat = LibrettoChat(self, self.config)
