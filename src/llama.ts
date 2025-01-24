import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { getLlama, LlamaChatSession, LlamaEmbedding } from "node-llama-cpp";
import path from "path";
import { fileURLToPath } from "url";
import { createIndexIfNotExists, index } from "./vectra.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const getEmbeddingContext = async () => {
  try {
    const llama = await getLlama();
    const model = await llama.loadModel({
      modelPath: path.join(
        __dirname,
        "../models",
        "hf_mradermacher_Llama-3.2-3B-Instruct.Q8_0.gguf"
      ),
    });

    const context = await model.createEmbeddingContext();
    return context;
  } catch (error) {
    const err = error as Error;
    throw new Error(`Failed to get embedding context: ${err.message}`);
  }
};

export const llamaChat = async (input: string) => {
  try {
    const llama = await getLlama();
    const model = await llama.loadModel({
      modelPath: path.join(
        __dirname,
        "../models",
        "hf_mradermacher_Llama-3.2-3B-Instruct.Q8_0.gguf"
      ),
    });

    const context = await model.createContext();
    const session = new LlamaChatSession({
      contextSequence: context.getSequence(),
    });

    const response = await session.prompt(input);
    return response;
  } catch (error) {
    const err = error as Error;
    throw new Error(`Failed to chat with Llama: ${err.message}`);
  }
};

// export const loadDocument = async (
//   documentPathParam: string
// ): Promise<string[]> => {
//   const documentPath = path.join(
//     __dirname,
//     "..",
//     "documents",
//     documentPathParam
//   );

//   try {
//     const data = await fs.readFile(documentPath, "utf-8");
//     console.log("data: ", data);
//     const chunks = await splitDocument(data);
//     return chunks;
//   } catch (error) {
//     const err = error as Error;
//     throw new Error(
//       `Failed to load document at path: ${documentPath} with a message error: ${err.message}`
//     );
//   }
// };

export const splitDocument = async (document: string): Promise<string[]> => {
  try {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500, // Increased chunk size
      chunkOverlap: 100,
    });
    const chunks = await splitter.splitText(document);
    //console.log("chunks: ", chunks);

    return chunks;
  } catch (error: any) {
    throw new Error(
      `Failed to split document with a message error: ${error.message}`
    );
  }
};

export const embedDocuments = async (documents: string[]): Promise<void> => {
  try {
    await createIndexIfNotExists();
    console.log("documents length: ", documents.length);

    const contextEmbedding = await getEmbeddingContext();

    await Promise.all(
      documents.map(async (document) => {
        //console.log("document: ", document);

        const embedding = await contextEmbedding.getEmbeddingFor(document);
        await index.insertItem({
          vector: [...embedding.vector],
          metadata: { text: document },
        });
      })
    );
  } catch (error) {
    const err = error as Error;
    throw new Error(`Failed to embed documents: ${err.message}`);
  }
};

export const getEmbeddingsForMessage = async (message: string) => {
  try {
    const contextEmbedding = await getEmbeddingContext();
    const embedding = await contextEmbedding.getEmbeddingFor(message);
    return embedding;
  } catch (error) {
    const err = error as Error;
    throw new Error(`Failed to get embeddings for message: ${err.message}`);
  }
};

export const findSimilarDocuments = async (
  embedding: LlamaEmbedding,
  documentEmbeddings: Map<string, LlamaEmbedding>
) => {
  const similarities = new Map<string, number>();
  for (const [otherDocument, otherDocumentEmbedding] of documentEmbeddings) {
    const similarity = embedding.calculateCosineSimilarity(
      otherDocumentEmbedding
    );
    // console.log(
    //   `Similarity between query and "${otherDocument}": ${similarity}`
    // );

    console.log(
      "*** Similarity ***",
      JSON.stringify(
        {
          otherDocument: otherDocument,
          similarity: similarity,
        },
        null,
        2
      )
    );
    similarities.set(otherDocument, similarity);
    // if (similarity > 0.4) {
    //   // Adjusted threshold
    //   // console.log(
    //   //   `Similarity between query and "${otherDocument}": ${similarity}`
    //   // );
    // }
  }

  return Array.from(similarities.keys()).sort(
    (a, b) => similarities.get(b)! - similarities.get(a)!
  );
};
