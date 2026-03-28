from hr_bot.generation.prompt import PromptBuilder


def test_prompt_includes_question():
    # Arrange
    builder = PromptBuilder()
    question = "What is the leave policy?"

    # Act
    system_prompt, user_prompt = builder.build(
        question=question,
        context="Some context"
    )

    # Assert
    assert question in user_prompt

def test_prompt_includes_context():
    # Arrange
    builder = PromptBuilder()
    context = "Employees get 20 days of leave."

    # Act
    system_prompt, user_prompt = builder.build(
        question="Leave?",
        context=context
    )

    # Assert
    assert context in user_prompt

def test_prompt_with_empty_context():
    # Arrange
    builder = PromptBuilder()

    # Act
    system_prompt, user_prompt = builder.build(
        question="Leave policy?",
        context=""
    )

    # Assert
    assert "Leave policy?" in user_prompt
    assert user_prompt is not None