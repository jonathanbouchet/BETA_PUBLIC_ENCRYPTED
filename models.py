from pydantic import BaseModel, Field, ValidationError
from typing import Union
from langchain.chains import create_tagging_chain_pydantic, create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from pydantic import Extra
from datetime import datetime


class Medication(BaseModel):
    """pydantic model for prescription
    """
    name: Union[str, None] = Field(description="the name of the prescription")
    dosage_quantity: Union[float, None] = Field(description="the quantity of the prescription")
    dosage_unit: Union[str, None] = Field(description="the unit of the prescription")
    usage: Union[str, None] = Field(description="the usage of the prescription")


class Tags0(BaseModel):
    """
    pydantic model for user's data
    """
    full_name: Union[str, None] = Field(description="this is the full name of the user", default=None)
    first_name: Union[str, None] = Field(description="this is the first name of the user", default=None)
    last_name: Union[str, None] = Field(description="this is the last name of the user", default=None)
    age: Union[int, None] = Field(description="this is the age of the user", default=None)
    date_of_birth: Union[str, None] = Field(description="this is the date of birth of the user", default=None)
    last_physician_seen: Union[str, None] = Field(description="this is the name of the last physician seen",
                                                  default=None)
    treatment_last_physician: Union[str, None] = Field(
        description="this is the treatment given during the last visit with the last physician seen",
        default=None)
    primary_doctor_name: Union[str, None] = Field(description="this is the name of the primary doctor", default=None)
    primary_doctor_specialty: Union[str, None] = Field(description="this is the specialty of the primary doctor",
                                                       default=None)
    primary_doctor_last_visit: Union[str, None] = Field(
        description="this is the date of the last visit with the primary doctor",
        default=None)
    weight: Union[float, None] = Field(description="this is the weight of the user", default=None)
    height: Union[float, None] = Field(description="this is the height of the user", default=None)
    weight_unit: Union[str, None] = Field(description="this is the unit of the weight of the user", default=None)
    height_unit: Union[str, None] = Field(description="this is the unit of the weight of the user", default=None)
    BMI: Union[float, None] = Field(description="this is the BMI of the user", default=None)
    had_weight_loss: Union[bool, None] = Field(description="did the user experience weight loss ?", default=None)
    weight_loss_reason: Union[str, None] = Field(description="if the user has lost weight, what is the reason ?",
                                                 default=None)
    weight_loss_value: Union[float, None] = Field(
        description="if the user has lost weight, what is the value of the loss?",
        default=None)
    is_taking_medications: Union[bool, None] = Field(
        description="did the user have prescription or non-prescription medications?", default=None)
    medications_details: list[Medication] = Field(description="medications taken by the user",
                                                  default=None)
    has_deformities: Union[bool, None] = Field(description="did the user have any amputations, physical deformities, "
                                                           "or received speech, physical, or occupational therapy in "
                                                           "the past 10 years?  ?",
                                               default=None)

    had_HIV: Union[bool, None] = Field(description="did the user have HIV in the last 10 years ?",
                                       default=None)
    is_pregnant: Union[bool, None] = Field(description="is the user pregnant ?", default=None)
    use_tobacco: Union[bool, None] = Field(description="does the user use tobacco or tobacco-related products?",
                                           default=None)
    use_alcohol: Union[bool, None] = Field(description="does the user consume alcohol?",
                                           default=None)
    used_marijuana: Union[bool, None] = Field(description="did the user used marijuana in the past 5 years",
                                              default=None)
    has_substance_counseling: Union[bool, None] = Field(description="did the user have or have been advised to have "
                                                                    "counseling or treatment for alcohol or drug use "
                                                                    "in the past 10 years?",
                                                        default=None)
    has_disability_benefits: Union[bool, None] = Field(
        description="did the user receive or apply for any disability benefits, including worker's compensation or "
                    "social security disability, in the past 5 years?",
        default=None)
    has_undisclosed_medical: Union[bool, None] = Field(
        description="did the user any undisclosed medical tests, exams, or scheduled appointments in the past 5 years?",
        default=None)
    has_family_history: Union[bool, None] = Field(
        description="did the user  Have any immediate family members (father, mother, sibling) died before age 60 due "
                    "to cardiovascular disease or cancer, or been diagnosed with diabetes, mental illness, "
                    "or hereditary conditions? ",
        default=None)

    class Config:
        extra = Extra.allow


def add_bmi(user_data: dict) -> dict:
    """calculate BMI as w / h*h where w = weight in kilograms and h = height in meter
    :params user_data: dict
    "returns: same user data with BMI added
    """
    height = user_data["height"]
    weight = user_data["weight"]
    if user_data["height_unit"] == "cm":
        height /= 100
    if user_data["height_unit"] == '"' or user_data["height_unit"] == "'":
        tmp = str(height).split(".")
        feet = int(tmp[0])
        inches = int(tmp[1])
        height = 2.54 * (feet*12 + inches) / 100
    if user_data["height_unit"] == "ft":
        height *= 30.48 / 100
    if user_data["weight_unit"] == "lbs":
        weight /= 2.205
    bmi = weight / (height * height)
    return bmi


def add_age(user_data: dict) -> dict:
    dob = user_data['date_of_birth']
    age = None
    if dob:
        try:
            dob_datetime = datetime.strptime(dob, "%m/%d/%Y")
            age = int(datetime.now().year - dob_datetime.year)
        except ValueError:
            try:
                dob_datetime = datetime.strptime(dob, "%m/%Y")
                age = int(datetime.now().year - dob_datetime.year)
            except ValueError:
                age = user_data["date_of_birth"]
    return age


def extract_data(assistant_summary: str, openai_key) -> dict:
    """
    run langchain tagging chain to extract data from summary according a pydantic model
    :param assistant_summary:
    :return:
    """

    llm_extraction = ChatOpenAI(temperature=0,
                                model="gpt-4",
                                openai_api_key=openai_key)

    # first we extract the summary data
    user_summary_extraction = create_extraction_chain_pydantic(Tags0, llm_extraction)
    extracted_data = user_summary_extraction.run(assistant_summary)
    # print(f"Pydantic data extracted:{extracted_data}")
    single_extracted_data = extracted_data[0]
    # print(f"single Pydantic data extracted:{extracted_data}")
    single_extracted_data_dict = dict(single_extracted_data)
    # print(f"data dict extracted:{single_extracted_data_dict}")
    current_bmi = add_bmi(single_extracted_data_dict)
    current_age = add_age(single_extracted_data_dict)
    single_extracted_data.age = current_age
    single_extracted_data.BMI = current_bmi

    return single_extracted_data


def check_extracted_data(extracted_data, query) -> None:
    """
    check the missing fields of the current response with respect a pydantic model
    :params extracted data:
    :returns: None
    """
    try:
        if len(extracted_data) == 0:
            print(f"no extracted data for the query: '{query}'")
        else:
            print(f"only partial extraction for the query: {extracted_data}")
    except ValidationError as e:
        print(f"validation missing errors: '{query}'")
        error_msg = e.errors()
        print(error_msg)


def add_non_empty_details(current_details, new_details):
    """
    add current fields extracted from user's query to existing pydantic model
    :params current_details:
    :params new_details:
    """
    new_details_dict = dict(new_details)
    non_empty_details = {k: v for k, v in new_details_dict.items() if v not in [None, "", 0]}
    updated_details = current_details.copy(update=non_empty_details)  # v1
    return updated_details


def filter_response(text_input: str, person, llm,):
    """
    search for fields of the current response with respect a pydantic model
    :params text_input: user's query
    :params person: pydantic model
    :params text_input: llm model for extraction
    """
    pydantic_model = Tags0

    chain = create_tagging_chain_pydantic(pydantic_model, llm)
    res = chain.run(text_input)
    print(f"current result :{res}, type:{type(res)}")

    person = add_non_empty_details(person, res)
    return person
