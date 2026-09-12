from sentence_transformers import SentenceTransformer
import time

start = time.perf_counter()
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

model_load_time = time.perf_counter()

documents = [
    """
    The user has been rebuilding their self-esteem and sense of self-worth after
    experiencing a painful romantic breakup. The experience caused them to question
    whether they were good enough, lovable enough, or somehow responsible for the
    relationship ending. Rejection, criticism, and comparisons with other people
    can sometimes trigger these insecurities. They are gradually trying to separate
    their sense of worth from whether another person chooses, validates, or approves
    of them.
    """,

    """
    The user is deeply interested in entrepreneurship and eventually wants to build
    something of their own. They are attracted not only to financial success but
    also to the autonomy, creative control, and responsibility involved in creating
    a company or product. A conventional career can be useful for gaining skills and
    experience, but the user does not necessarily see working for someone else as
    their ultimate destination.
    """,

    """
    The user is currently developing their technical abilities, particularly in
    machine learning, data structures and algorithms, programming, databases, and
    software development. They are relatively inexperienced with development and
    sometimes feel behind technically, but they deliberately want to build real
    systems rather than remain at a purely theoretical level. Their current
    technical learning is strongly connected to their career and placement goals.
    """,

    """
    The user strongly prefers deep, hands-on learning over blindly following
    tutorials or using abstractions that they cannot explain. When building a
    project, they want to understand what is happening underneath the frameworks,
    libraries, databases, and models they use. They are willing to struggle and move
    more slowly if that struggle results in genuine understanding and the ability
    to independently reproduce or explain the system.
    """,

    """
    The user is highly reflective and frequently uses philosophy, psychology,
    introspection, and ideas about human behaviour to understand their experiences.
    They tend to search for meaning and underlying principles rather than only
    looking for immediate solutions. Ideas concerning identity, resilience,
    impermanence, self-awareness, motivation, and personal growth are particularly
    meaningful to them.
    """,

    """
    The user is strongly motivated by personal growth and by the idea of becoming
    stronger through difficult experiences. They do not want painful periods,
    failures, rejection, or setbacks to permanently define them. They often find
    motivation in the belief that hardship can be transformed into strength,
    discipline, maturity, and a better version of themselves.
    """,

    """
    The user experiences anxiety and self-doubt when thinking about their future,
    particularly around academic performance, placements, technical competence,
    and whether they are progressing quickly enough. They sometimes compare their
    progress with peers and feel that they are falling behind. Despite this,
    career progress remains an important source of motivation and they actively try
    to improve their skills and opportunities.
    """,

    """
    The user values independence and autonomy in both their personal life and career.
    They want to feel that their decisions and direction belong to them rather than
    being controlled by external expectations or other people. Financial success is
    desirable, but freedom, control over their time and direction, and the ability
    to make their own choices are also important components of what they consider a
    successful life.
    """,

    """
    The user has a tendency to procrastinate or avoid difficult tasks when they do
    not understand how to begin or when a problem feels overwhelming. This is
    particularly noticeable with technically challenging work. However, they also
    recognize this pattern and want to become more disciplined and capable of
    working through uncertainty rather than escaping from difficult tasks.
    """,

    """
    The user has certain ideas, sentences, experiences, songs, and personal
    realizations that they find emotionally grounding during difficult periods.
    They sometimes want to be reminded of perspectives that previously helped them
    regain confidence, maintain perspective, or continue moving forward. These
    personally meaningful reminders can act as motivational anchors during moments
    of weakness or emotional difficulty.
    """
]
queries = [

    # ─────────────────────────────────────
    # SELF-WORTH / RELATIONSHIP / EMOTIONAL
    # ─────────────────────────────────────

    "After someone rejected me, I immediately started wondering whether there was something fundamentally wrong with me.",

    "I am slowly realizing that someone not choosing me does not necessarily say anything about my value as a person.",

    "The breakup made me question myself much more than I expected, especially whether I was actually good enough for someone.",

    "A negative comment about me triggered an insecurity that I thought I had already moved past.",

    "I don't want my confidence to depend on whether another person loves or validates me.",

    "I want to rebuild my identity without constantly measuring myself against the person I lost.",


    # ─────────────────────────────────────
    # ENTREPRENEURSHIP / AUTONOMY
    # ─────────────────────────────────────

    "I think I would eventually regret spending my entire professional life executing someone else's ideas.",

    "The idea of creating a company from scratch excites me because I would actually be responsible for deciding its direction.",

    "Money matters to me, but having control over what I do with my life matters almost as much.",

    "I want my career to eventually give me the freedom to choose what problems I work on.",

    "Getting a normal job could be useful right now, but I don't think I want employment to be the final destination.",


    # ─────────────────────────────────────
    # TECHNICAL LEARNING / DEVELOPMENT
    # ─────────────────────────────────────

    "I have been spending a lot of time learning machine learning and programming because I need stronger technical foundations.",

    "I know the theory of some technologies but I need to become capable of actually building systems with them.",

    "I am trying to become comfortable with databases, backend development, and machine learning instead of only knowing the concepts academically.",

    "My technical skills are still developing and I sometimes feel like I started seriously learning this stuff later than other people.",

    "The main reason I am building this project is that I want something substantial enough to prove that I can actually implement what I learn.",


    # ─────────────────────────────────────
    # DEEP LEARNING / HANDS-ON BUILDING
    # ─────────────────────────────────────

    "I could easily copy an implementation from the internet, but I would rather understand why every component exists.",

    "Using a library is not satisfying to me if I cannot explain what the library is doing underneath.",

    "I learn much more when I am forced to debug something myself instead of being shown the answer immediately.",

    "I would rather spend three hours understanding a system than finish it in thirty minutes without knowing how it works.",

    "I don't want a project that just lists ten technologies on my resume while I have no idea how those technologies actually interact.",


    # ─────────────────────────────────────
    # PHILOSOPHY / INTROSPECTION
    # ─────────────────────────────────────

    "When something painful happens, I usually end up thinking about what it reveals about me or what I can learn from it.",

    "I often try to understand the deeper reason behind my reactions instead of simply trying to suppress them.",

    "Philosophical ideas help me put temporary emotional experiences into a larger perspective.",

    "I spend a lot of time thinking about identity, human behaviour, and why people react to things the way they do.",

    "I find it useful to ask what a difficult experience means rather than only asking how to make the discomfort disappear.",


    # ─────────────────────────────────────
    # RESILIENCE / PERSONAL GROWTH
    # ─────────────────────────────────────

    "I don't want this difficult period to become the story of who I am forever.",

    "I want to come out of everything that happened stronger rather than permanently damaged by it.",

    "Even when something breaks me emotionally for a while, I want to use it to become more mature and resilient.",

    "One thing that keeps me going is the belief that today's weakness does not have to become tomorrow's identity.",

    "I want my failures and painful experiences to become things I grew from rather than things that define me.",


    # ─────────────────────────────────────
    # ANXIETY / COMPARISON / CAREER
    # ─────────────────────────────────────

    "With placements approaching, I sometimes feel like everyone else has progressed much further than I have.",

    "I get anxious when I think that I haven't learned enough to compete with other students.",

    "Sometimes I know I am improving but still feel behind because I compare my progress with people around me.",

    "My career uncertainty becomes much worse when I feel that I am running out of time.",

    "I want to become technically competent partly because I am afraid of being left behind professionally.",


    # ─────────────────────────────────────
    # PROCRASTINATION / DIFFICULTY
    # ─────────────────────────────────────

    "When I don't know how to start a difficult technical problem, I tend to avoid it instead of working through the uncertainty.",

    "Sometimes the hardest part of a task is getting myself to begin when I already expect it to be difficult.",

    "I have noticed that I can become distracted when a project reaches a part that I don't understand.",


    # ─────────────────────────────────────
    # MOTIVATIONAL ANCHORS
    # ─────────────────────────────────────

    "When I am having a weak moment, I want to be reminded that I have already survived difficult things before.",

    "There are certain ideas that I want my future self to hear again whenever I start doubting myself.",

    "A particular sentence or perspective can sometimes pull me out of a very negative state because I have personally found it meaningful before."
]

query_read_time = time.perf_counter()

query_embeddings = model.encode(queries)
document_embeddings = model.encode(documents)

embedding_time = time.perf_counter()

similarity_score = model.similarity(document_embeddings,query_embeddings)

similarity_search_time = time.perf_counter()

print(similarity_score)

print("-----------------------")
print(f"Model Loading Time: {model_load_time - start}")
print(f"Embedding Time: {embedding_time - query_read_time}")
print(f"Similarity Search Time: {similarity_search_time - embedding_time}")