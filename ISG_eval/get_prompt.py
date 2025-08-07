_EXTRACT_SEQUENCE_PROMPT = """Task: Extract key information from a multimodal prompt and format it into JSON.

Input: A prompt for a multimodal model to generate interleaved text-and-image content. The prompt may include input images.

Output: JSON format containing the following keys:
1. "Query": List representing the sequence of images and text in the input
2. "Answer": List representing the sequence of images and text in the expected output

Special Tokens:
- Input query:
  - Images: <query_img1>, <query_img2>, ...
  - Text: <query_text1>, <query_text2>, ...
- Generated output:
  - Images: <gen_img1>, <gen_img2>, ...
  - Text: <gen_text1>, <gen_text2>, ...

Example 1:
[Start of Example 1]
{example_2}
[End of Example 1]

Example 2:
[Start of Example 2]
{example_3}
[End of Example 2]


Instructions:
1. Analyze the given prompt to determine the number of images and text pieces to be generated.
2. Identify the sequence of images and text in both the query and the expected answer.
3. Format the extracted information into the specified JSON structure.
4. There will not be adjacent <gen_text>, such as <gen_text{{X}}> <gen_text{{X+1}}>. 
5. Only output the sequence of images and text noted by <gen_text> and <gen_img> in the "Query" and "Answer".
6. Think before you output your final answer, you can format your thought in a key "Thought" in your output and explain your answer to be generated.

Here is the prompt: 
[Start of Prompt]
{input_prompt}
[End of Prompt]
"""

_EXTRACT_RELATION_PROMPT = """Task: Extract and format the relationships between elements in a multimodal prompt and its expected generated answer.

Input:
1. Original prompt for a multimodal model
2. Sequence of elements represented by special tokens

Special Tokens:
- Input query:
  - Images: <query_img1>, <query_img2>, ...
  - Text: <query_text1>, <query_text2>, ...
- Generated output:
  - Images: <gen_img1>, <gen_img2>, ...
  - Text: <gen_text1>, <gen_text2>, ...
- Whole answer: <all>

Output: JSON format with a "relation" key containing a list of triplets.

Relation Triplet Format: (<subject>, <object>, <relation>)
- <relation> is an open-vocabulary description (phrase or short sentence)
- The triplet should be able to form a fluent English sentence: <subject> <relation> <object>
- Avoid duplicate triplets
- Only include relations explicitly described in the prompt
- The order of <subject> and <object> should reflect the most logical and fluent relationship, regardless of their sequence in the input or output

Instructions:
1. Analyze the given prompt carefully.
2. Identify explicit relationships between elements in both the prompt and expected answer.
3. Format relationships as triplets according to the specified format.
4. Ensure the triplets can be easily constructed into fluent English sentences.
5. Use specific descriptors for relations, forming phrases or short sentences.
6. Ensure all triplets are unique and explicitly described in the prompt.
7. Order <subject> and <object> in each triplet to create the most logical and fluent relationship. Do not include relations between input images and texts.
8. Compile the triplets into a list under the "relation" key in the JSON output.
9. Think before you output your final answer, you can format your thought in a key "Thought" in your output and explain your answer to be generated.


Example 1:
[Start of Example 1]
Input prompt: "Create a van Gogh style variant of the input image <query_img1>. Then, decompose this variant into 4 individual objects. For each object, generate an image and provide a brief caption. Present the results as: [Variant image] [Variant description], followed by [Object1 image] [Object1 caption], [Object2 image] [Object2 caption], and so on."

Element sequence: {{
  "Query": [
      "<query_img1>",
      "<query_text1>"
  ],
  "Answer": [
      "<gen_text1>",
      "<gen_img1>",
      "<gen_text2>",
      "<gen_img2>",
      "<gen_text3>",
      "<gen_img3>",
      "<gen_text4>",
      "<gen_img4>",
      "<gen_text5>"
  ]
}}
                     
Expected JSON output:
{{
  "relation": [
    ("<gen_img1>", "<query_img1>", "is a van Gogh style variant of"),
    ("<gen_text1>", "<gen_img1>", "is a description of"),
    ("<gen_img2>", "<gen_img1>", "is a part of"),
    ("<gen_img3>", "<gen_img1>", "is a part of"),
    ("<gen_img4>", "<gen_img1>", "is a part of"),
    ("<gen_img5>", "<gen_img1>", "is a part of"),
    ("<gen_text2>", "<gen_img2>", "is a caption for"),
    ("<gen_text3>", "<gen_img3>", "is a caption for"),
    ("<gen_text4>", "<gen_img4>", "is a caption for"),
    ("<gen_text5>", "<gen_img5>", "is a caption for")
  ]
}}
[End of Example 1]

Example 2:
[Start of Example 2]
{example_2}
[End of Example 2]

Example 3:
[Start of Example 3]
{example_3}
[End of Example 3]

Note: This example includes only the relations that can be confidently inferred from the prompt. The triplets are formed to create fluent English sentences when read as "<subject> <relation> <object>". Do not include any obscure or ambiguous relations that can't be understood by only reading the triplet. For example, "is an image of the third object extracted from" is an obscure relation because there is no "third" within this triplet, you can use "is an image of the object extracted from" instead.

Here is the prompt:
[Start of Prompt]
{input_prompt}
[End of Prompt]

Here is the element sequence:
[Start of Element Sequence]
{input_element_sequence}
[End of Element Sequence]
"""

_EXTRACT_TUPLE_PROMPT = """Task: Predict atomic concrete visual entities, attributes, and relations for generated images based on a prompt for a multimodal generative model.

Input:
1. Original prompt for a multimodal generative model
2. Sequence of elements represented by special tokens

Special Tokens:
- Input query:
  - Images: <query_img1>, <query_img2>, ...
  - Text: <query_text1>, <query_text2>, ...
- Generated output:
  - Images: <gen_img1>, <gen_img2>, ...
  - Text: <gen_text1>, <gen_text2>, ...

Output: 
JSON format with a "tuple" key containing a list of tuples in the following formats:
1. Entity: (entity, name of entity, image_id), for example: ["entity", "fish", "<gen_img1>"]
2. Attribute: (attribute, name of attribute, entity, image_id), for example: ["attribute", "yellow", "fish", "<gen_img1>"]
3. Relation: (relation, name of relation, entity1, entity2, image_id), for example: ["relation", "swim in", "fish", "water", "<gen_img1>"]


Example 1:
[Start of Example 1]
{example_2}
[End of Example 1]

Example 2:
[Start of Example 2]
{example_3}
[End of Example 2]


Instructions:
1. Carefully analyze the given prompt, focusing on predicting concrete, visual elements that are highly likely to appear in the generated images. Do not describe or analyze any input images mentioned in the prompt.
2. If the prompt includes descriptions or captions of multiple input images, identify common themes and key visual elements across these descriptions. Use these commonalities to inform your predictions for the generated images.
3. Predict tangible entities first. These should be physical objects or beings that can be visually represented in generated images. Avoid abstract concepts or general scenes like 'scene', 'atmosphere', or 'landscape'.
4. For each predicted entity, identify its likely visual attributes. Focus on characteristics that would be visibly apparent in a generated image.
5. You should atomize the entity and attribute as much as possible, and generate entity tuple first, then attribute tuple, and finally relation tuple. Make sure the entities, attributes, and relations are atomic. For example, you should not output "yellow fish" as an entity, you should output "fish" as an entity and "yellow" as an attribute.
6. For attributes and relations, always reference the specific entity or entities they are associated with.
8. Specify which generated image (image_id) each predicted element is expected to appear in. If an entity is likely to appear in multiple generated images, create separate tuples for each image. DO NOT INCLUDE tuples can't be inferred from prompt.
9. Only include tuples for elements you are highly confident (100% Sure) will appear in the generated images based on the prompt and common sense reasoning. Avoid speculating about details that aren't strongly implied by the prompt!
10. For prompts describing a sequence of generated images, consider how visual elements might change or interact across the sequence.
11. Compile the tuples into a list under the "tuple" key in the JSON output.

Requirements: 
1. Follow the example! 
2. {task_specific_note}

Here is the prompt:
[Start of Prompt]
{input_prompt}
[End of Prompt]

Here is the element sequence:
[Start of Element Sequence]
{input_element_sequence}
[End of Element Sequence]
"""

_EXTRACT_VL_TUPLE_PROMPT = """Task: Predict atomic concrete visual entities, attributes, and relations for generated images based on a prompt and images for a multimodal generative model.

Input:
1. Original prompt for a multimodal generative model
2. Sequence of elements represented by special tokens
3. Images mentionedin the prompt


Output: 
JSON format with a "tuple" key containing a list of tuples in the following formats:
1. Entity: (entity, name of entity, image_id), for example: ["entity", "fish", "<gen_img1>"]
2. Attribute: (attribute, name of attribute, entity, image_id), for example: ["attribute", "yellow", "fish", "<gen_img1>"]
3. Relation: (relation, name of relation, entity1, entity2, image_id), for example: ["relation", "under", "woman", "umbrella", "<gen_img2>"]


Example 1:
[Start of Example 1]
{example_2}
[End of Example 1]

Example 2:
[Start of Example 2]
{example_3}
[End of Example 2]


Instructions:
1. Carefully analyze the given prompt, focusing on predicting concrete, visual elements that are highly likely to appear in the generated images. Do not describe or analyze any input images mentioned in the prompt.
2. If the prompt includes descriptions or captions of multiple input images, identify common themes and key visual elements across these descriptions. Use these commonalities to inform your predictions for the generated images.
3. Predict tangible entities first. These should be physical objects or beings that can be visually represented in generated images. Avoid abstract concepts or general scenes like 'scene', 'atmosphere', or 'landscape'.
4. For each predicted entity, identify its likely visual attributes. Focus on characteristics that would be visibly apparent in a generated image.
5. You should atomize the entity and attribute as much as possible, and generate entity tuple first, then attribute tuple, and finally relation tuple. Make sure the entities, attributes, and relations are atomic. For example, you should not output "yellow fish" as an entity, you should output "fish" as an entity and "yellow" as an attribute.
6. For attributes and relations, always reference the specific entity or entities they are associated with.
8. Specify which generated image (image_id) each predicted element is expected to appear in. If an entity is likely to appear in multiple generated images, create separate tuples for each image. DO NOT INCLUDE tuples can't be inferred from prompt.
9. Only include tuples for elements you are highly confident (100% Sure) will appear in the generated images based on the prompt and common sense reasoning. Avoid speculating about details that aren't strongly implied by the prompt!
10. For prompts describing a sequence of generated images, consider how visual elements might change or interact across the sequence.
11. Compile the tuples into a list under the "tuple" key in the JSON output.

Requirements: 
1. Follow the example!
2. {task_specific_note}

Here is the prompt:
[Start of Prompt]
{input_prompt}
[End of Prompt]

Here is the element sequence:
[Start of Element Sequence]
{input_element_sequence}
[End of Element Sequence]
"""

_GENERATE_RELATION_QA_PROMPT = """Task: Create questions for each provided triplet to verify the stated relationship.


Special Tokens:
- Input query:
  - Images: <query_img1>, <query_img2>, ...
  - Text: <query_text1>, <query_text2>, ...
- Generated output:
  - Images: <gen_img1>, <gen_img2>, ...
  - Text: <gen_text1>, <gen_text2>, ...
- Whole answer: <all>


Input: A list of triplets in the format (<subject>, <object>, <relation>).

Output: A JSON list of objects, each containing the original triplet information and a generated question.

Example 1:
[Start of Example 1]
{example_2}
[End of Example 1]

Example 2:
[Start of Example 2]
{example_3}
[End of Example 2]


Note: Ensure that the generated questions are diverse in their phrasing while maintaining clarity and relevance to the original triplet. The questions should be designed to elicit a yes/no or true/false response that verifies the relationship stated in the triplet. Remember to replace image references with "first image", "second image", etc., but keep text references as they are.

Instructions:
1. For each input triplet, create an object with the following structure:
   {{
     "subject": "<subject from triplet>",
     "object": "<object from triplet>",
     "relation": "relation from triplet",
     "Question": "<generated question>"
   }}
2. Generate a question that, when answered, would verify whether the stated relationship between the subject and object is correct.
3. Ensure the question is clear, concise, and directly related to the triplet's content.
4. Replace image references (e.g., <gen_img1>, <query_img1>) with "this image" if only one image occurs in the triplet, otherwise replace with "first image", "second image" for the subject and object based on their order of appearance in the triplet. 
5. Do not use "third" or "fourth" images in the question, as in the question, the maximum number of images could only be 2 (subject and object).
6. Keep text references (e.g., <gen_text1>, <query_text1>) as they are in the original triplet.
7. Frame the question in a way that can be answered with a yes/no or true/false response.
8. Compile all generated objects into a JSON list.

Notice: If subject and object are images (<gen_img1>, <query_img1>, etc.), refer the first image as "first image" and the second image as "second image" in your generated question.

Here is the input:
[Start of Input]
{input_prompt}
[End of Input]
"""

_GENERATE_ATTRIBUTE_QA_PROMPT = """Task: Create questions for each provided triplet of entity, attribute, or relation, and format them into a specific JSON structure.

Special Tokens:
- Input query:
  - Images: <query_img1>, <query_img2>, ...
  - Text: <query_text1>, <query_text2>, ...
- Generated output:
  - Images: <gen_img1>, <gen_img2>, ...
  - Text: <gen_text1>, <gen_text2>, ...
- Whole answer: <all>

Input: A list of triplets in the following formats:
1. Entity: (entity, name of entity, image_id)
2. Attribute: (attribute, name of attribute, entity, image_id)
3. Relation: (relation, name of relation, entity1, entity2, image_id)

Output: A JSON list of objects, each containing the generated question and related information.

Example 1:
[Start of Example 1]
{example_2}
[End of Example 1]

Example 2:
[Start of Example 2]
{example_3}
[End of Example 2]


Instructions:
1. For each input triplet, create an object with the following structure:
   {{
     "image": special token refer to generated images,
     "Question": "<generated question>",
     "id": <numeric id starting from 0>,
     "Preliminary": [<list of prerequisite question ids>]
   }}
2. Generate a question that verifies the existence of the entity, the presence of the attribute, or the relationship between entities.
3. Ensure the question is clear, concise, and can be answered with a yes/no response.
4. Assign a unique numeric id to each question, starting from 0.
5. Determine any prerequisite questions and list their ids in the "Preliminary" field.
   - For attributes, include the id of the corresponding entity question.
   - For relations, include the ids of both entity questions.
6. Compile all generated objects into a JSON list.


Note: Ensure that the generated questions are clear and directly related to the triplet's content. The "Preliminary" field should accurately reflect the dependencies between questions, especially for attributes and relations that depend on the existence of entities.

Here is the input:
[Start of Input]
{input_prompt}
[End of Input]
"""

gen_requirement = {
  "Detailed_Requirement": [
    "Semantic Segmentation",
    "3D_object",
    "Scene Attribute Transfer",
    "multi-Perspective Scene Generation",
    "Portrait Variation",
    "Art Style Transfer",
    "Photo Variation",
    "Imaginary Object Detection",
    "Realistic Object Detection"
  ],
  "Partly_Requirement": [
    "Animation Text-Image",
    "Animation Images",
    "Attribute Transfer",
    "Prediction",
    "painting",
    "visual storytelling text-image",
    "visual storytelling text",
    "visual storytelling image",
    "HowTo"
  ],
  "No_Requirement": [
    "Historical",
    "Object_VQA",
    "Scientific"
  ]
}

task_specific_note = {
  "Detailed_Requirement": "This is an accurate generative task, which means that the predicted generative content is definitive. You must output any entities, attributes, and relations that can be clearly inferred from the prompt, as shown in the example.",
  "Partly_Requirement": "You can only output the main and core entities, attributes, and relations that can be 100% sure inferred from the prompt, as shown in the example.",
  "No_Requirement": "You don't need to output anything."
}

import json
import os


def get_prompt():
  with open("./ISG_eval/task_few_shot.json", "r", encoding="utf-8") as f:
    data = json.load(f)
  prompt_dict = {
    "extract_sequence": _EXTRACT_SEQUENCE_PROMPT,
    "extract_relation": _EXTRACT_RELATION_PROMPT,
    "extract_tuple": _EXTRACT_TUPLE_PROMPT,
    "extract_vl_tuple": _EXTRACT_VL_TUPLE_PROMPT,
    "generate_relation_qa": _GENERATE_RELATION_QA_PROMPT,
    "generate_attribute_qa": _GENERATE_ATTRIBUTE_QA_PROMPT,
    "gen_requirement": gen_requirement,
    "task_specific_note": task_specific_note
  }
  return data, prompt_dict