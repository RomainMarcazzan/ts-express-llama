import express, { Request, Response } from "express";
import multer from "multer";
import cors from "cors";
import pdf from "pdf-parse";

import {
  embedDocuments,
  findSimilarDocuments,
  getEmbeddingsForMessage,
  llamaChat,
  splitDocument,
} from "./llama.js";
import {
  getVectorDBContents,
  resetDatabase,
  retrieveDocumentEmbeddingsFromDb,
} from "./vectra.js";

const app = express();
const upload = multer();
app.use(cors());

app.use(express.json());

app.get("/health", (req: Request, res: Response) => {
  res.send("OK");
});

app.post("/chat-trained", async (req: Request, res: Response) => {
  try {
    const { message } = req.body;
    if (!message) {
      throw new Error("Message is required");
    }
    const documentEmbeddingsFromDb = await retrieveDocumentEmbeddingsFromDb();
    const queryEmbedding = await getEmbeddingsForMessage(message);
    const similarDocuments = await findSimilarDocuments(
      queryEmbedding,
      documentEmbeddingsFromDb
    );

    const topDocuments = similarDocuments.slice(0, 3);
    const context = `Documents similaires : ${topDocuments.join(", ")}`;

    //console.log("context", context);

    const customPrompt = `
    Utilisez les informations suivantes pour répondre à la question de l'utilisateur.

    Contexte : ${context}
    Question : ${message}

    Si la question n'est pas en relation avec le contexte, répondez du mieux possible sans tenir compte du contexte.
    `;

    const response = await llamaChat(customPrompt);
    res.json({ response });
  } catch (error) {
    const err = error as Error;
    res.status(500).json({ error: err.message });
  }
});

app.post("/chat", async (req: Request, res: Response) => {
  try {
    const { message } = req.body;
    if (!message) {
      throw new Error("Message is required");
    }
    const response = await llamaChat(message);
    res.json({ ai: response });
  } catch (error) {
    const err = error as Error;
    res.status(500).json({ error: err.message });
  }
});

app.post(
  "/embed",
  upload.single("document"),
  async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        throw new Error("Document file is required");
      }
      const fileBuffer = req.file.buffer;
      const fileType = req.file.mimetype;

      if (!fileBuffer) {
        throw new Error("Document file is required");
      }

      let documentContent: string;

      if (fileType === "application/pdf") {
        const pdfData = await pdf(fileBuffer);
        documentContent = pdfData.text;
      } else if (fileType === "text/plain") {
        documentContent = fileBuffer.toString("utf-8");
      } else {
        throw new Error("Unsupported file type");
      }
      const cleanData = documentContent.replace(/\n/g, " ").trim();

      console.log("documentContent: ", documentContent);

      const chunks = await splitDocument(cleanData);
      await embedDocuments(chunks);
      res.status(200).json({ message: "Document embedded successfully" });
    } catch (error) {
      const err = error as Error;
      res.status(500).json({ error: err.message });
    }
  }
);

app.get("/visualize-db", async (req: Request, res: Response) => {
  try {
    const items = await getVectorDBContents();
    res.status(200).json({ items });
  } catch (error) {
    const err = error as Error;
    res.status(500).json({ error: err.message });
  }
});

app.delete("/reset-db", async (req: Request, res: Response) => {
  try {
    await resetDatabase();
    res.status(200).json({ message: "Vectra DB reset successfully" });
  } catch (error) {
    const err = error as Error;
    res.status(500).json({ error: err.message });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
