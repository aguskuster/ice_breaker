from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

information = """
Masashi Kishimoto nació en la prefectura de Okayama, Japón el 8 de noviembre de 1974, junto a su hermano gemelo menor, Seishi Kishimoto. Durante su infancia, Kishimoto mostró interés por los personajes de dibujo animado.

Fue en este momento de su vida en que Masashi Kishimoto comenzó a pensar que el manga era genial y deseaba convertirse en un famoso mangaka como Akira Toriyama creando su primer manga cerca de este momento y que llevaba por título "Hiatari-kun", una historia que giraba acerca de “un niño ninja de las sombras”.

En la escuela intermedia, Kishimoto comenzó a enfocarse en otras cosas diferentes al dibujo. El béisbol se convirtió en gran parte de su vida, y naturalmente tenía que dedicar más tiempo al estudio, lo que significó poco o ningún tiempo para dibujar. Se comenzó a preocupar de si estaba “muy viejo para dibujar”, y en este punto un evento increíble ocurrió en su vida.
"""
if __name__ == "__main__":
    print("Hello LangChain")

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=information))
