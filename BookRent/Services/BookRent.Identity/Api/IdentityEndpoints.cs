namespace BookRent.Identity.Api;

public static class IdentityEndpoints
{
    public static void MapIdentityEndPoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/identity");

        group.MapGet("/login", Login);
        group.MapPost("/register", Register);
        group.MapGet("/role/{id:guid}", GetUserRole);
    }


    private static IResult Login(HttpContext context)
    {
        return Results.Ok();
    }
    
    private static IResult Register(HttpContext context)
    {
        return Results.Ok();
    }
    
    private static IResult GetUserRole(HttpContext context)
    {
        return Results.Ok();
    }
}