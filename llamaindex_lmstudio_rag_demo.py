import requests
from typing import Any, Sequence, AsyncGenerator, Generator

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import LLM, CompletionResponse, LLMMetadata, ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class MyLLM(LLM):
    model_name: str
    api_base: str
    api_key: str

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=512,
            model_name=self.model_name,
            is_chat_model=False,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': kwargs.get('temperature', 0.7),
        }
        response = requests.post(f'{self.api_base}/v1/chat/completions', json=payload, headers=headers)
        response.raise_for_status()
        text = response.json()['choices'][0]['message']['content']
        return CompletionResponse(text=text)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponse:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        # 👇 ChatMessage を dict に変換
        openai_messages = [
            {'role': m.role.value, 'content': m.content} for m in messages
        ]

        payload = {
            'model': self.model_name,
            'messages': openai_messages,
            'temperature': kwargs.get('temperature', 0.7),
        }

        response = requests.post(
            f'{self.api_base}/v1/chat/completions',
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']

        chat_message = ChatMessage(role='assistant', content=content)
        return CompletionResponse(text=content, message=chat_message)

    # ストリーミングと非同期は未対応の場合、NotImplementedErrorでOK
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        raise NotImplementedError('stream_complete is not implemented.')

    def stream_chat(self, messages: Sequence[dict], **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        raise NotImplementedError('stream_chat is not implemented.')

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError('acomplete is not implemented.')

    async def achat(self, messages: Sequence[dict], **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError('achat is not implemented.')

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        raise NotImplementedError('astream_complete is not implemented.')

    async def astream_chat(self, messages: Sequence[dict], **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        raise NotImplementedError('astream_chat is not implemented.')


def chat(query_engine):
    print()
    print('チャットモード開始 (終了するには "quit" と入力)')
    print('-' * 30)

    while True:
        try:
            print()
            user_input = input('あなた: ').strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print('終了します。')
                break

            if not user_input:
                continue

            # クエリ実行
            response = query_engine.query(user_input)

            # 結果表示
            print()
            print('🧠 LLMからの回答:')
            print(response.response)

        except KeyboardInterrupt:
            print('\n終了します。')
            break
        except Exception as e:
            print(f'エラーが発生しました: {str(e)}')


def main():
    # ✅ LM Studio で起動しているモデル名を指定
    llm = MyLLM(
        model_name='llama-3-elyza-jp-8b',
        api_base='http://localhost:1234',
        api_key='lm-studio',
    )

    # 埋め込みモデルは HuggingFace の日本語対応モデル
    embed_model = HuggingFaceEmbedding(
        model_name='sonoisa/sentence-bert-base-ja-mean-tokens-v2',
        device='cpu',
    )
    # embed_model = HuggingFaceEmbedding(model_name='cl-tohoku/bert-base-japanese')  # ←fugashi必須

    # 文書読み込み
    documents = SimpleDirectoryReader('./rag_data').load_data()

    # インデックス作成（llmは渡さない）
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # LLM付きクエリエンジンを作成
    query_engine = index.as_query_engine(llm=llm)

    chat(query_engine=query_engine)
    # # クエリ実行
    # response = query_engine.query('津市の鰻屋さんを3つ紹介してください。')

    # # 結果表示
    # print('🧠 LLMからの回答:')
    # print(response.response)


if __name__ == '__main__':
    main()
