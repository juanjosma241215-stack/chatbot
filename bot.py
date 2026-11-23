import json
import os
import re
import logging
import torch
import torch.nn.functional as F
from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
UMBRAL_SIMILITUD = 0.45
IMAGEN_MENU = os.path.join("imagen", "image.png")

SALUDOS_PREFIJO = (
    "hola", "buenas", "buenos dias", "buenos días",
    "buenas tardes", "buenas noches", "hey", "holi", "disculpa", "ola"
)

print("Cargando modelo de lenguaje de El Bodegón...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def limpiar_texto(texto: str) -> str:
    t = texto.lower().strip()
    t = re.sub(r"[¿?¡!.,]", " ", t)
    palabras = t.split()
    while palabras and palabras[0] in SALUDOS_PREFIJO:
        palabras.pop(0)
    if len(palabras) >= 2 and f"{palabras[0]} {palabras[1]}" in SALUDOS_PREFIJO:
        palabras = palabras[2:]
    return " ".join(palabras)


def obtener_embedding(textos: list[str]) -> torch.Tensor:
    encoded = tokenizer(
        textos, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**encoded)
    token_embeddings = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return F.normalize(embeddings, p=2, dim=1)


with open("faqs.json", "r", encoding="utf-8") as f:
    FAQS = json.load(f)

TEXTOS_INDEXADOS = [
    f"{item.get('mostrar', '')} {item['pregunta']}" for item in FAQS
]
EMBEDDINGS_FAQS = obtener_embedding(TEXTOS_INDEXADOS)
print(f"Carta de preguntas de El Bodegón cargada con {len(FAQS)} temas.")


def buscar_faq(consulta_usuario: str) -> dict | None:
    limpio = limpiar_texto(consulta_usuario)

    if not limpio:
        return None

    emb_usuario = obtener_embedding([limpio])
    similitudes = torch.mm(emb_usuario, EMBEDDINGS_FAQS.transpose(0, 1))[0]

    palabras_usuario = {p for p in limpio.split() if len(p) > 3}
    for idx, item in enumerate(FAQS):
        texto_faq = f"{item['pregunta']} {item.get('mostrar', '')}".lower()
        if any(p in texto_faq for p in palabras_usuario):
            similitudes[idx] += 0.15

    mejor_score, mejor_idx = torch.max(similitudes, dim=0)

    if mejor_score.item() >= UMBRAL_SIMILITUD:
        return FAQS[mejor_idx.item()]

    return None


def es_consulta_menu(texto: str) -> bool:
    limpio = limpiar_texto(texto)
    palabras_menu = {"menu", "menú", "carta", "venden", "platos", "hamburguesas", "comida"}
    return any(p in limpio.split() for p in palabras_menu)


async def responder_seguro(update: Update, texto: str, usar_foto_menu: bool = False):
    try:
        if usar_foto_menu and os.path.exists(IMAGEN_MENU):
            with open(IMAGEN_MENU, "rb") as foto:
                await update.message.reply_photo(
                    photo=foto,
                    caption=texto,
                    parse_mode="Markdown",
                )
        else:
            await update.message.reply_text(texto, parse_mode="Markdown")
    except (TimedOut, NetworkError) as e:
        logging.warning(f"Problema temporal de red al enviar respuesta: {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await responder_seguro(
        update,
        "🍔🍻 *¡Bienvenido a El Bodegón!*\n\n"
        "Soy tu anfitrión virtual. Pregúntame lo que necesites con tus propias palabras "
        "(por ejemplo: _¿qué venden?_, _horario_, _domicilios_, _reservas_ o _parqueadero_).\n\n"
        "También puedes usar /preguntas para ver todo lo que te puedo contar.",
    )


async def listar_preguntas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lista = "\n".join(f"• {item.get('mostrar', item['pregunta'])}" for item in FAQS)
    await responder_seguro(
        update,
        f"📋 *Preguntas frecuentes de El Bodegón:*\n\n{lista}",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    consulta = update.message.text
    limpio = limpiar_texto(consulta)

    if not limpio:
        await responder_seguro(
            update,
            "¡Hola! 👋 Bienvenido a *El Bodegón* 🍔🍻\n"
            "¿En qué te puedo ayudar hoy? Puedes preguntarme por nuestro *horario, carta, cócteles, reservas, domicilios o parqueadero*.",
        )
        return

    faq_encontrada = buscar_faq(consulta)

    if faq_encontrada:
        respuesta = faq_encontrada["respuesta"]
        es_menu = (
            "menu" in faq_encontrada.get("pregunta", "").lower()
            or "carta" in faq_encontrada.get("pregunta", "").lower()
            or es_consulta_menu(consulta)
        )
        await responder_seguro(update, respuesta, usar_foto_menu=es_menu)
        return

    await responder_seguro(
        update,
        "¡Ups! No estoy seguro de haber entendido esa consulta 😅\n"
        "En *El Bodegón* puedo ayudarte con información sobre:\n"
        "• Horarios y ubicación\n"
        "• Menú, hamburguesas y cócteles\n"
        "• Reservas y cumpleaños\n"
        "• Domicilios, parqueadero y pagos\n\n"
        "Escribe /preguntas para ver todas las consultas frecuentes.",
    )


def main():
    if not TOKEN:
        raise ValueError("Falta la variable de entorno TELEGRAM_TOKEN.")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .connect_timeout(20.0)
        .read_timeout(20.0)
        .write_timeout(20.0)
        .get_updates_connect_timeout(20.0)
        .get_updates_read_timeout(20.0)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("preguntas", listar_preguntas))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("El Bodegón Bot ejecutándose... Presiona Ctrl+C para detenerlo.")
    app.run_polling(bootstrap_retries=5)


if __name__ == "__main__":
    main()